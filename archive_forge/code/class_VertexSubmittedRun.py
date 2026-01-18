import asyncio
import logging
from typing import Any, Dict, Optional
from wandb.apis.internal import Api
from wandb.util import get_module
from .._project_spec import LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..environment.gcp_environment import GcpEnvironment
from ..errors import LaunchError
from ..registry.abstract import AbstractRegistry
from ..utils import MAX_ENV_LENGTHS, PROJECT_SYNCHRONOUS, event_loop_thread_exec
from .abstract import AbstractRun, AbstractRunner, Status
class VertexSubmittedRun(AbstractRun):

    def __init__(self, job: Any) -> None:
        self._job = job

    @property
    def id(self) -> str:
        return self._job.name

    async def get_logs(self) -> Optional[str]:
        return None

    @property
    def name(self) -> str:
        return self._job.display_name

    @property
    def gcp_region(self) -> str:
        return self._job.location

    @property
    def gcp_project(self) -> str:
        return self._job.project

    def get_page_link(self) -> str:
        return '{console_uri}/vertex-ai/locations/{region}/training/{job_id}?project={project}'.format(console_uri=GCP_CONSOLE_URI, region=self.gcp_region, job_id=self.id, project=self.gcp_project)

    async def wait(self) -> bool:
        await self._job.wait()
        return (await self.get_status()).state == 'finished'

    async def get_status(self) -> Status:
        job_state = str(self._job.state)
        if job_state == 'JobState.JOB_STATE_SUCCEEDED':
            return Status('finished')
        if job_state == 'JobState.JOB_STATE_FAILED':
            return Status('failed')
        if job_state == 'JobState.JOB_STATE_RUNNING':
            return Status('running')
        if job_state == 'JobState.JOB_STATE_PENDING':
            return Status('starting')
        return Status('unknown')

    async def cancel(self) -> None:
        self._job.cancel()