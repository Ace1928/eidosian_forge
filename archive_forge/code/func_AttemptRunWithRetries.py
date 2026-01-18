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
def AttemptRunWithRetries(command_name, worker, exit_statuses, cmd, env, output_file, multiple_workers, run_cmd):
    """Attempts to connect to a worker using SSH or SCP."""
    max_attempts = 10
    sleep_interval = 5
    for i in range(max_attempts):
        try:
            log.status.Print('{}: Attempting to connect to worker {}...'.format(command_name, worker))
            exit_status = run_cmd(env, cmd, output_file)
            if exit_status:
                if multiple_workers:
                    log.status.Print('##### Command execution on worker {} failed with exit status {}. Continuing.'.format(worker, exit_status))
                    exit_statuses[worker] = exit_status
                sys.exit(exit_status)
        except ssh.CommandError as e:
            if i == max_attempts - 1:
                if multiple_workers:
                    exit_statuses[worker] = 255
                raise e
            if multiple_workers:
                log.status.Print('Failed to execute command on multiple workers. This may have happened if you have not added your SSH key to your ssh-agent using "ssh-add ~/.ssh/google_compute_engine".')
            log.status.Print('Retrying: {} command error: {}'.format(command_name, six.text_type(e)))
            time.sleep(sleep_interval)
            continue
        break