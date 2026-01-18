from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def PrepareNodeForSSH(tpu_name, user, args, release_track, enable_batching, username_requested, prepped_nodes, idx):
    """Prepares TPU VM Node for SSH.

  Args:
    tpu_name: The unqualified name of the tpu instance to prepare for SSH.
    user: The username to use for SSH.
    args: The command line arguments used for SSH.
    release_track: The release track/version of client protos to use.
    enable_batching: A bool, True if the user opts into batching requests.
    username_requested: A bool, True if the user has passed a specific username
      in the args.
    prepped_nodes: The list to put resulting prepared nodes into.
    idx: The index specifying which slot in the 'prepped_nodes' list to put the
      output node.

  Returns:
    The prepared node that is now ready for SSH.

  Raises:
    BadArgumentException: If the node retrieved is not a TPU VM. Non TPU VMs are
      not supported.
    InvalidArgumentException: If there is no command specified, but multiple
      workers are targeted.
    IapTunnelingUnavailable: If IAP Tunneling is not available, the node cannot
      be SSHed into.
  """
    prepped_node = tpu_utils.SSHPreppedNode(tpu_name, user, release_track, enable_batching)
    tpu = tpu_utils.TPUNode(release_track)
    node = tpu.Get(tpu_name, args.zone)
    if not tpu_utils.IsTPUVMNode(node):
        raise exceptions.BadArgumentException('TPU', 'this command is only available for Cloud TPU VM nodes. To access this node, please see https://cloud.google.com/tpu/docs/creating-deleting-tpus.')
    ValidateTPUState(node.state, tpu.messages.Node.StateValueValuesEnum, tpu_name)
    prepped_node.worker_ips = ParseWorkerFlag(args.worker, node.networkEndpoints, args.internal_ip)
    if len(prepped_node.worker_ips) > 1 and (not args.command):
        raise exceptions.InvalidArgumentException('--worker', 'cannot target multiple workers without the `--command` flag.')
    if node.health == tpu.messages.Node.HealthValueValuesEnum.UNHEALTHY_MAINTENANCE:
        log.warning('!!! This TPU is going through a maintenance event, and might be unavailable !!!')
    single_pod_worker = len(node.networkEndpoints) > 1 and len(prepped_node.worker_ips) == 1
    guest_attributes_response = GetGuestAttributes(tpu, single_pod_worker, prepped_node.worker_ips, tpu_name, args.zone)
    if guest_attributes_response is None:
        if args.IsKnownAndSpecified('tunnel_through_iap') and args.tunnel_through_iap:
            log.debug('Unable to retrieve host information from guest attributes.')
            log.status.Print('Failed to connect to TPU.')
            log.status.Print(IAP_TROUBLESHOOTING_HELP)
            raise tpu_exceptions.IapTunnelingUnavailable()
        log.debug('Unable to retrieve host keys from guest attributes. Continuing.')
        prepped_node.host_key_suffixes = None
    else:
        prepped_node.host_key_suffixes = GetHostKeySuffixesFromGuestAttributes(guest_attributes_response, single_pod_worker, prepped_node.worker_ips, node)
    prepped_node.ssh_helper = ssh_utils.BaseSSHCLIHelper()
    prepped_node.ssh_helper.Run(args)
    public_key = prepped_node.ssh_helper.keys.GetPublicKey().ToEntry()
    prepped_node.project = tpu_utils.GetProject(release_track, prepped_node.ssh_helper)
    if not args.plain:
        _, expiration_micros = ssh_utils.GetSSHKeyExpirationFromArgs(args)
        oslogin_state = ssh.GetOsloginState(None, prepped_node.project, user, public_key, expiration_micros, release_track, username_requested=username_requested, instance_enable_oslogin=TpuHasOsLoginEnabled(node), messages=base_classes.ComputeApiHolder(release_track).client.messages)
        prepped_node.user = user = oslogin_state.user
    public_key = '{1}:{0} {1}'.format(public_key, user)
    if not args.plain and (not args.dry_run):
        AddSSHKeyIfNeeded(prepped_node.project, tpu, node, tpu_name, args.zone, public_key)
    prepped_node.command_list = args.command.split(' ') if args.command else None
    prepped_node.remainder = []
    if args.ssh_args:
        prepped_node.remainder.extend(args.ssh_args)
    output_directory_path = None
    if args.output_directory:
        log.status.Print('Preparing SSH command execution; output will be logged to {}'.format(output_directory_path))
    prepped_node.instance_names = {}
    if args.IsKnownAndSpecified('tunnel_through_iap') and args.tunnel_through_iap:
        for worker in prepped_node.worker_ips:
            index = 0 if single_pod_worker else worker
            instance_name = GetFromGuestAttributes(guest_attributes_response.guestAttributes, index, 'hostname')
            if instance_name is None:
                log.status.Print('Failed to connect to TPU.')
                log.status.Print(IAP_TROUBLESHOOTING_HELP)
                raise tpu_exceptions.IapTunnelingUnavailable()
            prepped_node.instance_names[worker] = instance_name
    prepped_node.id = node.id
    prepped_nodes[idx] = prepped_node
    return prepped_node