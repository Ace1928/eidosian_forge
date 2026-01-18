import asyncio
import logging
from typing import Any, Dict, List, Optional, cast
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.environment.aws_environment import AwsEnvironment
from wandb.sdk.launch.errors import LaunchError
from .._project_spec import EntryPoint, LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..registry.abstract import AbstractRegistry
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
class SagemakerSubmittedRun(AbstractRun):
    """Instance of ``AbstractRun`` corresponding to a subprocess launched to run an entry point command on aws sagemaker."""

    def __init__(self, training_job_name: str, client: 'boto3.Client', log_client: Optional['boto3.Client']=None) -> None:
        super().__init__()
        self.client = client
        self.log_client = log_client
        self.training_job_name = training_job_name
        self._status = Status('running')

    @property
    def id(self) -> str:
        return f'sagemaker-{self.training_job_name}'

    async def get_logs(self) -> Optional[str]:
        if self.log_client is None:
            return None
        try:
            describe_log_streams = event_loop_thread_exec(self.log_client.describe_log_streams)
            describe_res = await describe_log_streams(logGroupName='/aws/sagemaker/TrainingJobs', logStreamNamePrefix=self.training_job_name)
            if len(describe_res['logStreams']) == 0:
                wandb.termwarn(f'Failed to get logs for training job: {self.training_job_name}')
                return None
            log_name = describe_res['logStreams'][0]['logStreamName']
            get_log_events = event_loop_thread_exec(self.log_client.get_log_events)
            res = await get_log_events(logGroupName='/aws/sagemaker/TrainingJobs', logStreamName=log_name)
            return '\n'.join([f'{event['timestamp']}:{event['message']}' for event in res['events']])
        except self.log_client.exceptions.ResourceNotFoundException:
            wandb.termwarn(f'Failed to get logs for training job: {self.training_job_name}')
            return None
        except Exception as e:
            wandb.termwarn(f'Failed to handle logs for training job: {self.training_job_name} with error {str(e)}')
            return None

    async def wait(self) -> bool:
        while True:
            status_state = (await self.get_status()).state
            wandb.termlog(f'{LOG_PREFIX}Training job {self.training_job_name} status: {status_state}')
            if status_state in ['stopped', 'failed', 'finished']:
                break
            await asyncio.sleep(5)
        return status_state == 'finished'

    async def cancel(self) -> None:
        status = await self.get_status()
        if status.state == 'running':
            self.client.stop_training_job(TrainingJobName=self.training_job_name)
            await self.wait()

    async def get_status(self) -> Status:
        describe_training_job = event_loop_thread_exec(self.client.describe_training_job)
        job_status = (await describe_training_job(TrainingJobName=self.training_job_name))['TrainingJobStatus']
        if job_status == 'Completed' or job_status == 'Stopped':
            self._status = Status('finished')
        elif job_status == 'Failed':
            self._status = Status('failed')
        elif job_status == 'Stopping':
            self._status = Status('stopping')
        elif job_status == 'InProgress':
            self._status = Status('running')
        return self._status