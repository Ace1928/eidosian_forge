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
def SSHPrepCmd(args, prepped_node, worker, ips):
    """Prepares the SSH command used to SSH into the worker.

  Args:
    args: The arguments passed in by the user.
    prepped_node: The object that contains all the necessary information to ssh
      into the node.
    worker: the worker to ssh into.
    ips: The ips of the worker

  Returns:
    ssh.SSHCommand that can be used to execute SSH command.
  """
    identity_file = None
    options = None
    if not args.plain:
        identity_file = prepped_node.ssh_helper.keys.key_file
        options = prepped_node.ssh_helper.GetConfig(GetInstanceID(prepped_node.id, worker, prepped_node.host_key_suffixes), args.strict_host_key_checking, None)
    remote = ssh.Remote(ips.ip_address, prepped_node.user)
    extra_flags = ssh.ParseAndSubstituteSSHFlags(args, remote, ips.ip_address, ips.internal_address)
    iap_tunnel_args = None
    if args.IsKnownAndSpecified('tunnel_through_iap') and args.tunnel_through_iap:
        instance_name = prepped_node.instance_names[worker]
        iap_tunnel_args = CreateSshTunnelArgs(args, prepped_node.release_track, prepped_node.project, args.zone, instance_name)
    return ssh.SSHCommand(remote=remote, identity_file=identity_file, remote_command=prepped_node.command_list, extra_flags=extra_flags, options=options, remainder=prepped_node.remainder, iap_tunnel_args=iap_tunnel_args)