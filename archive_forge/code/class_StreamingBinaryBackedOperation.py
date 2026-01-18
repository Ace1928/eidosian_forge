from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class StreamingBinaryBackedOperation(six.with_metaclass(abc.ABCMeta, BinaryBackedOperation)):
    """Extend Binary Operations for binaries which require streaming output."""

    def __init__(self, binary, binary_version=None, check_hidden=False, std_out_func=None, std_err_func=None, failure_func=None, default_args=None, custom_errors=None, capture_output=False, structured_output=False, install_if_missing=False):
        super(StreamingBinaryBackedOperation, self).__init__(binary, binary_version, check_hidden, std_out_func, std_err_func, failure_func, default_args, custom_errors, install_if_missing)
        self.capture_output = capture_output
        if structured_output:
            default_out_handler = DefaultStreamStructuredOutHandler
            default_err_handler = DefaultStreamStructuredErrHandler
        else:
            default_out_handler = DefaultStreamOutHandler
            default_err_handler = DefaultStreamErrHandler
        self.std_out_handler = std_out_func or default_out_handler
        self.std_err_handler = std_err_func or default_err_handler
        self.structured_output = structured_output

    def _Execute(self, cmd, stdin=None, env=None, **kwargs):
        """Execute binary and return operation result.

     Will parse args from kwargs into a list of args to pass to underlying
     binary and then attempt to execute it. Will use configured stdout, stderr
     and failure handlers for this operation if configured or module defaults.

    Args:
      cmd: [str], command to be executed with args
      stdin: str, data to send to binary on stdin
      env: {str, str}, environment vars to send to binary.
      **kwargs: mapping of additional arguments to pass to the underlying
        executor.

    Returns:
      OperationResult: execution result for this invocation of the binary.

    Raises:
      ArgumentError, if there is an error parsing the supplied arguments.
      BinaryOperationError, if there is an error executing the binary.
    """
        op_context = {'env': env, 'stdin': stdin, 'exec_dir': kwargs.get('execution_dir')}
        result_holder = self.OperationResult(command_str=cmd, execution_context=op_context)
        std_out_handler = self.std_out_handler(result_holder=result_holder, capture_output=self.capture_output)
        std_err_handler = self.std_err_handler(result_holder=result_holder, capture_output=self.capture_output)
        short_cmd_name = os.path.basename(cmd[0])
        try:
            working_dir = kwargs.get('execution_dir')
            if working_dir and (not os.path.isdir(working_dir)):
                raise InvalidWorkingDirectoryError(short_cmd_name, working_dir)
            exit_code = execution_utils.ExecWithStreamingOutput(args=cmd, no_exit=True, out_func=std_out_handler, err_func=std_err_handler, in_str=stdin, cwd=working_dir, env=env)
        except (execution_utils.PermissionError, execution_utils.InvalidCommandError) as e:
            raise ExecutionError(short_cmd_name, e)
        result_holder.exit_code = exit_code
        self.set_failure_status(result_holder, kwargs.get('show_exec_error', False))
        return result_holder