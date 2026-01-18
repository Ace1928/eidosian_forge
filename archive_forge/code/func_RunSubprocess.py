from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import subprocess
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
import six
def RunSubprocess(proc_name, command_list):
    """Runs a subprocess and prints out the output.

  Args:
    proc_name: The name of the subprocess to call.
      Used for error logging.
    command_list: A list with all the arguments for a subprocess call.
      Must be able to be passed to a subprocess.Popen call.
  """
    try:
        proc = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for l in iter(proc.stdout.readline, ''):
            log.out.write(l)
            log.out.flush()
        proc.wait()
        if proc.returncode != 0:
            raise OSError(proc.stderr.read().strip())
    except OSError as e:
        log.err.Print('Error running %s: %s' % (proc_name, six.text_type(e)))
        command_list_str = ' '.join(command_list)
        log.err.Print('INVOCATION: %s' % command_list_str)