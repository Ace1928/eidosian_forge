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
def SCPIntoPreppedNodes(prepped_nodes, args, total_scp_batch_size):
    """SCP's into the prepped nodes.

  Args:
    prepped_nodes: The list of prepared nodes to be SCPed into.
    args: The list of arguments passed in to SCP with.
    total_scp_batch_size: The final parsed batch size to SCP into the nodes'
      workers.
  """
    scp_threads = []
    current_batch_size = 0
    worker_ips = []
    for prepped_node in prepped_nodes:
        worker_ips.extend(prepped_node.worker_ips.items())
    num_ips = len(worker_ips)
    exit_statuses = [None] * num_ips
    log.status.Print('Using scp batch size of {}.Attempting to SCP into {} nodes with a total of {} workers.'.format(total_scp_batch_size, len(prepped_nodes), num_ips))
    for prepped_node in prepped_nodes:
        for worker, ips in prepped_node.worker_ips.items():
            cmd = SCPPrepCmd(args, prepped_node, worker, ips)
            if args.dry_run:
                log.out.Print(' '.join(cmd.Build(prepped_node.ssh_helper.env)))
                continue
            if len(prepped_node.worker_ips) > 1 or len(prepped_nodes) > 1:
                scp_threads.append(threading.Thread(target=AttemptRunWithRetries, args=('SCP', worker, exit_statuses, cmd, prepped_node.ssh_helper.env, None, True, SCPRunCmd)))
                scp_threads[-1].start()
                current_batch_size += 1
                if prepped_node.enable_batching and current_batch_size == total_scp_batch_size:
                    WaitForBatchCompletion(scp_threads, exit_statuses)
                    current_batch_size = 0
                    scp_threads = []
            else:
                AttemptRunWithRetries('SCP', worker, exit_statuses, cmd, prepped_node.ssh_helper.env, None, False, SCPRunCmd)
    if len(worker_ips) > 1 and scp_threads:
        WaitForBatchCompletion(scp_threads, exit_statuses)