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
class BinaryBackedOperation(six.with_metaclass(abc.ABCMeta, object)):
    """Class for declarative operations implemented as external binaries."""

    class OperationResult(object):
        """Generic Holder for Operation return values and errors."""

        def __init__(self, command_str, output=None, errors=None, status=0, failed=False, execution_context=None):
            self.executed_command = command_str
            self.stdout = output
            self.stderr = errors
            self.exit_code = status
            self.context = execution_context
            self.failed = failed

        def __str__(self):
            output = collections.OrderedDict()
            output['executed_command'] = self.executed_command
            output['stdout'] = self.stdout
            output['stderr'] = self.stderr
            output['exit_code'] = self.exit_code
            output['failed'] = self.failed
            output['execution_context'] = self.context
            return yaml.dump(output)

        def __eq__(self, other):
            if isinstance(other, BinaryBackedOperation.OperationResult):
                return self.executed_command == other.executed_command and self.stdout == other.stdout and (self.stderr == other.stderr) and (self.exit_code == other.exit_code) and (self.failed == other.failed) and (self.context == other.context)
            return False

        def __repr__(self):
            return self.__str__()

    def __init__(self, binary, binary_version=None, check_hidden=False, std_out_func=None, std_err_func=None, failure_func=None, default_args=None, custom_errors=None, install_if_missing=False):
        """Creates the Binary Operation.

    Args:
      binary: executable, the name of binary containing the underlying
        operations that this class will invoke.
      binary_version: string, version of the wrapped binary.
      check_hidden: bool, whether to look for the binary in hidden components.
      std_out_func: callable(OperationResult, **kwargs), returns a function to
        call to process stdout from executable and build OperationResult
      std_err_func: callable(OperationResult, **kwargs), returns a function to
        call to process stderr from executable and build OperationResult
      failure_func: callable(OperationResult), function to call to determine if
        the operation result is a failure. Useful for cases where underlying
        binary can exit with non-zero error code yet still succeed.
      default_args: dict{str:str}, mapping of parameter names to values
        containing default/static values that should always be passed to the
        command.
      custom_errors: dict(str:str}, map of custom exception messages to be used
        for known errors.
      install_if_missing: bool, if True prompt for install on missing component.
    """
        self._executable = CheckForInstalledBinary(binary_name=binary, check_hidden=check_hidden, install_if_missing=install_if_missing, custom_message=custom_errors['MISSING_EXEC'] if custom_errors else None)
        self._binary = binary
        self._version = binary_version
        self._default_args = default_args
        self.std_out_handler = std_out_func or DefaultStdOutHandler
        self.std_err_handler = std_err_func or DefaultStdErrHandler
        self.set_failure_status = failure_func or DefaultFailureHandler

    @property
    def binary_name(self):
        return self._binary

    @property
    def executable(self):
        return self._executable

    @property
    def defaults(self):
        return self._default_args

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
        std_out_handler = self.std_out_handler(result_holder)
        std_err_handler = self.std_err_handler(result_holder)
        short_cmd_name = os.path.basename(cmd[0])
        try:
            working_dir = kwargs.get('execution_dir')
            if working_dir and (not os.path.isdir(working_dir)):
                raise InvalidWorkingDirectoryError(short_cmd_name, working_dir)
            exit_code = execution_utils.Exec(args=cmd, no_exit=True, out_func=std_out_handler, err_func=std_err_handler, in_str=stdin, cwd=working_dir, env=env)
        except (execution_utils.PermissionError, execution_utils.InvalidCommandError) as e:
            raise ExecutionError(short_cmd_name, e)
        result_holder.exit_code = exit_code
        self.set_failure_status(result_holder, kwargs.get('show_exec_error', False))
        return result_holder

    @abc.abstractmethod
    def _ParseArgsForCommand(self, **kwargs):
        """Parse and validate kwargs into command argument list.

    Will process any default_args first before processing kwargs, overriding as
    needed. Will also perform any validation on passed arguments. If calling a
    named sub-command on the underlying binary (vs. just executing the root
    binary), the sub-command should be the 1st argument returned in the list.

    Args:
      **kwargs: keyword arguments for the underlying command.

    Returns:
     list of arguments to pass to execution of underlying command.

    Raises:
      ArgumentError: if there is an error parsing or validating arguments.
    """
        pass

    def __call__(self, **kwargs):
        cmd = [self.executable]
        cmd.extend(self._ParseArgsForCommand(**kwargs))
        return self._Execute(cmd, **kwargs)