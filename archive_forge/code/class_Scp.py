from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
import time
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import ssh as qr_ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import util as queued_resource_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Scp(base.Command):
    """Copy files to and from a Cloud TPU Queued Resource via SCP."""
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
        tpu_ssh_utils.AddTPUSSHArgs(parser, cls._ENABLE_IAP, cls._ENABLE_BATCHING, enable_batching_default=cls.DEFAULT_BATCH_SIZE)
        AddSCPArgs(parser)
        flags.AddZoneFlag(parser, resource_type='tpu', operation_type='scp')

    def Run(self, args):
        start_time = time.time()
        dst = ssh.FileReference.FromPath(args.destination)
        srcs = [ssh.FileReference.FromPath(src) for src in args.sources]
        ssh.SCPCommand.Verify(srcs, dst, single_remote=True)
        remote = dst.remote or srcs[0].remote
        qr_name = remote.host
        if not dst.remote:
            for src in srcs:
                src.remote = remote
        username_requested = True
        if not remote.user:
            username_requested = False
            remote.user = ssh.GetDefaultSshUsername(warn_on_account_user=True)
        if args.zone is None:
            args.zone = properties.VALUES.compute.zone.Get(required=True)
        queued_resource_client = queued_resource_utils.TPUQueuedResource(self.ReleaseTrack())
        queued_resource = queued_resource_client.Get(qr_name, args.zone)
        node_specs = qr_ssh_utils.ParseNodeFlag(args.node, queued_resource.tpu.nodeSpec)
        if len(node_specs) > 1 and srcs[0].remote:
            raise exceptions.InvalidArgumentException('--node', 'cannot target multiple nodes while copying files to client.')
        prep_nodes_threads = []
        current_batch_size = 0
        num_nodes = len(node_specs)
        prep_node_batch_size = tpu_ssh_utils.ParseBatchSize(args.batch_size, len(node_specs))
        prepped_nodes = [None] * num_nodes
        for index, node in enumerate(node_specs):
            prep_nodes_threads.append(threading.Thread(target=tpu_ssh_utils.PrepareNodeForSCP, args=(node.nodeId, args, self.ReleaseTrack(), self._ENABLE_BATCHING, username_requested, prepped_nodes, index, srcs, dst, remote)))
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
        scp_batch_size = tpu_ssh_utils.ParseBatchSize(args.batch_size, self.DEFAULT_BATCH_SIZE)
        tpu_ssh_utils.SCPIntoPreppedNodes(prepped_nodes, args, scp_batch_size)
        log.status.Print('Completed execution in %s seconds' % (time.time() - start_time))