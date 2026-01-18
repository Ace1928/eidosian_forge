from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def ExecWithStreamingOutputNonThreaded(args, env=None, no_exit=False, out_func=None, err_func=None, in_str=None, raw_output=False, **extra_popen_kwargs):
    """Emulates the os.exec* set of commands, but uses subprocess.

  This executes the given command, waits for it to finish, and then exits this
  process with the exit code of the child process. Allows realtime processing of
  stderr and stdout from subprocess without threads.

  Args:
    args: [str], The arguments to execute.  The first argument is the command.
    env: {str: str}, An optional environment for the child process.
    no_exit: bool, True to just return the exit code of the child instead of
      exiting.
    out_func: str->None, a function to call with each line of the stdout of the
      executed process. This can be e.g. log.file_only_logger.debug or
      log.out.write.
    err_func: str->None, a function to call with each line of the stderr of
      the executed process. This can be e.g. log.file_only_logger.debug or
      log.err.write.
    in_str: bytes or str, input to send to the subprocess' stdin.
    raw_output: bool, stream raw lines of output perserving line
      endings/formatting.
    **extra_popen_kwargs: Any additional kwargs will be passed through directly
      to subprocess.Popen

  Returns:
    int, The exit code of the child if no_exit is True, else this method does
    not return.

  Raises:
    PermissionError: if user does not have execute permission for cloud sdk bin
    files.
    InvalidCommandError: if the command entered cannot be found.
  """
    log.debug('Executing command: %s', args)
    env = GetToolEnv(env=env)
    process_holder = _ProcessHolder()
    with _ReplaceSignal(signal.SIGTERM, process_holder.Handler):
        with _ReplaceSignal(signal.SIGINT, process_holder.Handler):
            out_handler_func = out_func or log.Print
            err_handler_func = err_func or log.status.Print
            if in_str:
                extra_popen_kwargs['stdin'] = subprocess.PIPE
            try:
                if args and isinstance(args, list):
                    args = [encoding.Encode(a) for a in args]
                p = subprocess.Popen(args, env=env, stderr=subprocess.PIPE, stdout=subprocess.PIPE, **extra_popen_kwargs)
                if in_str:
                    in_str = six.text_type(in_str).encode('utf-8')
                    try:
                        p.stdin.write(in_str)
                        p.stdin.close()
                    except OSError as exc:
                        if exc.errno == errno.EPIPE or exc.errno == errno.EINVAL:
                            pass
                        else:
                            _KillProcIfRunning(p)
                            raise OutputStreamProcessingException(exc)
                try:
                    _StreamSubprocessOutput(p, stdout_handler=out_handler_func, stderr_handler=err_handler_func, raw=raw_output)
                except Exception as e:
                    _KillProcIfRunning(p)
                    raise OutputStreamProcessingException(e)
            except OSError as err:
                if err.errno == errno.EACCES:
                    raise PermissionError(err.strerror)
                elif err.errno == errno.ENOENT:
                    raise InvalidCommandError(args[0])
                raise
            process_holder.process = p
            if process_holder.signum is not None:
                _KillProcIfRunning(p)
            ret_val = p.returncode
    if no_exit and process_holder.signum is None:
        return ret_val
    sys.exit(ret_val)