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
def SCPPrepCmd(args, prepped_node, worker, ips):
    """Prepares the SCP command used to SCP into the worker.

  Args:
    args: The arguments passed in by the user.
    prepped_node: The object that contains all the necessary information to scp
      into the node.
    worker: the worker to scp into.
    ips: The ips of the worker

  Returns:
    ssh.SCPCommand that can be used to execute SCP command.
  """
    worker_dst = copy.deepcopy(prepped_node.dst)
    if worker_dst.remote:
        worker_dst.remote.host = ips.ip_address
    else:
        prepped_node.srcs[0].remote.host = ips.ip_address
    options = None
    if not args.plain:
        options = prepped_node.ssh_helper.GetConfig(GetInstanceID(prepped_node.id, worker, prepped_node.host_key_suffixes), args.strict_host_key_checking, None)
    iap_tunnel_args = None
    if args.IsKnownAndSpecified('tunnel_through_iap') and args.tunnel_through_iap:
        instance_name = prepped_node.instance_names[worker]
        iap_tunnel_args = CreateSshTunnelArgs(args, prepped_node.release_track, prepped_node.project, args.zone, instance_name)
    return ssh.SCPCommand(prepped_node.srcs, worker_dst, identity_file=prepped_node.identity_file, options=options, recursive=args.recurse, compress=args.compress, extra_flags=prepped_node.extra_flags, iap_tunnel_args=iap_tunnel_args)