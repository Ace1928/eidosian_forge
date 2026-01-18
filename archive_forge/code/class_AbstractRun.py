import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dockerpycreds.utils import find_executable  # type: ignore
import wandb
from wandb.apis.internal import Api
from wandb.sdk.lib import runid
from .._project_spec import LaunchProject
class AbstractRun(ABC):
    """Wrapper around a W&B launch run.

    A launched run is a subprocess running an entry point
    command, that exposes methods for waiting on and cancelling the run.
    This class defines the interface that the W&B launch runner uses to manage the lifecycle
    of runs launched in different environments (e.g. runs launched locally or in a cluster).
    ``AbstractRun`` is not thread-safe. That is, concurrent calls to wait() / cancel()
    from multiple threads may inadvertently kill resources (e.g. local processes) unrelated to the
    run.
    """

    def __init__(self) -> None:
        self._status = Status()

    @property
    def status(self) -> Status:
        return self._status

    @abstractmethod
    async def get_logs(self) -> Optional[str]:
        """Return the logs associated with the run."""
        pass

    def _run_cmd(self, cmd: List[str], output_only: Optional[bool]=False) -> Optional[Union['subprocess.Popen[bytes]', bytes]]:
        """Run the command and returns a popen object or the stdout of the command.

        Arguments:
        cmd: The command to run
        output_only: If true just return the stdout bytes
        """
        try:
            env = os.environ
            popen = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE)
            if output_only:
                popen.wait()
                if popen.stdout is not None:
                    return popen.stdout.read()
            return popen
        except subprocess.CalledProcessError as e:
            wandb.termerror(f'Command failed: {e}')
            return None

    @abstractmethod
    async def wait(self) -> bool:
        """Wait for the run to finish, returning True if the run succeeded and false otherwise.

        Note that in some cases, we may wait until the remote job completes rather than until the W&B run completes.
        """
        pass

    @abstractmethod
    async def get_status(self) -> Status:
        """Get status of the run."""
        pass

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the run (interrupts the command subprocess, cancels the run, etc).

        Cancels the run and waits for it to terminate. The W&B run status may not be
        set correctly upon run cancellation.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> Optional[str]:
        pass