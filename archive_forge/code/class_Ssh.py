from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os.path
import threading
import time
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import ssh as qr_ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import util as queued_resource_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Ssh(base.Command):
    """Send SSH commands to a Cloud TPU Queued Resource."""
    _ENABLE_IAP = True
    _ENABLE_BATCHING = True
    DEFAULT_BATCH_SIZE = 64

    @classmethod
    def Args(cls, parser):
        """Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    """
        ssh_utils.BaseSSHCLIHelper.Args(parser)
        AddSSHArgs(parser)
        tpu_ssh_utils.AddTPUSSHArgs(parser, enable_iap=cls._ENABLE_IAP, enable_batching=cls._ENABLE_BATCHING, enable_batching_default=cls.DEFAULT_BATCH_SIZE)
        AddCommandArgGroup(parser)
        flags.AddZoneFlag(parser, resource_type='tpu', operation_type='ssh')

    def Run(self, args):
        start_time = time.time()
        user, qr_name = ssh_utils.GetUserAndInstance(args.user_queued_resource)
        if args.zone is None:
            args.zone = properties.VALUES.compute.zone.Get(required=True)
        if args.output_directory:
            output_directory_path = os.path.abspath(os.path.expandvars(os.path.expanduser(args.output_directory)))
            if not os.path.isdir(output_directory_path):
                raise exceptions.InvalidArgumentException('--output_directory', 'Failed to find directory {}. Please create it or specify another directory'.format(output_directory_path))
        queued_resource_client = queued_resource_utils.TPUQueuedResource(self.ReleaseTrack())
        queued_resource = queued_resource_client.Get(qr_name, args.zone)
        username_requested = '@' in args.user_queued_resource
        node_specs = qr_ssh_utils.ParseNodeFlag(args.node, queued_resource.tpu.nodeSpec)
        prep_nodes_threads = []
        current_batch_size = 0
        num_nodes = len(node_specs)
        prep_node_batch_size = tpu_ssh_utils.ParseBatchSize(args.batch_size, len(node_specs))
        prepped_nodes = [None] * num_nodes
        for index, node in enumerate(node_specs):
            prep_nodes_threads.append(threading.Thread(target=tpu_ssh_utils.PrepareNodeForSSH, args=(node.nodeId, user, args, self.ReleaseTrack(), self._ENABLE_BATCHING, username_requested, prepped_nodes, index)))
            prep_nodes_threads[-1].start()
            current_batch_size += 1
            if current_batch_size == prep_node_batch_size:
                qr_ssh_utils.WaitForNodeBatchCompletion(prep_nodes_threads, prepped_nodes)
                current_batch_size = 0
                prep_nodes_threads = []
        if current_batch_size > 0:
            qr_ssh_utils.WaitForNodeBatchCompletion(prep_nodes_threads, prepped_nodes)
        prepped_nodes = [prepped_node for prepped_node in prepped_nodes if prepped_node is not None]
        if len(prepped_nodes) < num_nodes:
            log.warning('Could not prepare all {} nodes, attempting to ssh into the rest.'.format(num_nodes))
        ssh_batch_size = tpu_ssh_utils.ParseBatchSize(args.batch_size, self.DEFAULT_BATCH_SIZE)
        tpu_ssh_utils.SSHIntoPreppedNodes(prepped_nodes, args, ssh_batch_size)
        log.status.Print('Completed execution in %s seconds' % (time.time() - start_time))