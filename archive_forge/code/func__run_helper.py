import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from getpass import getuser
from shlex import quote
from typing import Dict, List
import click
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.docker import (
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.subprocess_output_util import (
from ray.autoscaler.command_runner import CommandRunnerInterface
def _run_helper(self, final_cmd, with_output=False, exit_on_fail=False, silent=False):
    """Run a command that was already setup with SSH and `bash` settings.

        Args:
            cmd (List[str]):
                Full command to run. Should include SSH options and other
                processing that we do.
            with_output (bool):
                If `with_output` is `True`, command stdout will be captured and
                returned.
            exit_on_fail (bool):
                If `exit_on_fail` is `True`, the process will exit
                if the command fails (exits with a code other than 0).

        Raises:
            ProcessRunnerError if using new log style and disabled
                login shells.
            click.ClickException if using login shells.
        """
    try:
        if not with_output:
            return run_cmd_redirected(final_cmd, process_runner=self.process_runner, silent=silent, use_login_shells=is_using_login_shells())
        else:
            return self.process_runner.check_output(final_cmd)
    except subprocess.CalledProcessError as e:
        joined_cmd = ' '.join(final_cmd)
        if not is_using_login_shells():
            raise ProcessRunnerError('Command failed', 'ssh_command_failed', code=e.returncode, command=joined_cmd)
        if exit_on_fail:
            raise click.ClickException('Command failed:\n\n  {}\n'.format(joined_cmd)) from None
        else:
            fail_msg = 'SSH command failed.'
            if is_output_redirected():
                fail_msg += ' See above for the output from the failure.'
            raise click.ClickException(fail_msg) from None
    finally:
        sys.stdout.flush()
        sys.stderr.flush()