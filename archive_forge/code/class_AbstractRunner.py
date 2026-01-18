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
class AbstractRunner(ABC):
    """Abstract plugin class defining the interface needed to execute W&B Launches.

    You can define subclasses of ``AbstractRunner`` and expose them as third-party
    plugins to enable running W&B projects against custom execution backends
    (e.g. to run projects against your team's in-house cluster or job scheduler).
    """
    _type: str

    def __init__(self, api: Api, backend_config: Dict[str, Any]) -> None:
        self._api = api
        self.backend_config = backend_config
        self._cwd = os.getcwd()
        self._namespace = runid.generate_id()

    def find_executable(self, cmd: str) -> Any:
        """Cross platform utility for checking if a program is available."""
        return find_executable(cmd)

    @property
    def api_key(self) -> Any:
        return self._api.api_key

    def verify(self) -> bool:
        """This is called on first boot to verify the needed commands, and permissions are available.

        For now just call `wandb.termerror` and `sys.exit(1)`
        """
        if self._api.api_key is None:
            wandb.termerror("Couldn't find W&B api key, run wandb login or set WANDB_API_KEY")
            sys.exit(1)
        return True

    @abstractmethod
    async def run(self, launch_project: LaunchProject, image_uri: str) -> Optional[AbstractRun]:
        """Submit an LaunchProject to be run.

        Returns a SubmittedRun object to track the execution
        Arguments:
        launch_project: Object of _project_spec.LaunchProject class representing a wandb launch project

        Returns:
            A :py:class:`wandb.sdk.launch.runners.SubmittedRun`. This function is expected to run
            the project asynchronously, i.e. it should trigger project execution and then
            immediately return a `SubmittedRun` to track execution status.
        """
        pass