import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import yaml
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.agent.agent import LaunchAgent
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from wandb.sdk.launch.registry.azure_container_registry import AzureContainerRegistry
from wandb.sdk.launch.registry.local_registry import LocalRegistry
from wandb.sdk.launch.runner.abstract import Status
from wandb.sdk.launch.runner.kubernetes_monitor import (
from wandb.util import get_module
from .._project_spec import EntryPoint, LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner
import kubernetes_asyncio  # type: ignore # noqa: E402
from kubernetes_asyncio import client  # noqa: E402
from kubernetes_asyncio.client.api.batch_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.core_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.custom_objects_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.models.v1_secret import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.rest import ApiException  # type: ignore # noqa: E402
class KubernetesSubmittedRun(AbstractRun):
    """Wrapper for a launched run on Kubernetes."""

    def __init__(self, batch_api: 'BatchV1Api', core_api: 'CoreV1Api', name: str, namespace: Optional[str]='default', secret: Optional['V1Secret']=None) -> None:
        """Initialize a KubernetesSubmittedRun.

        Other implementations of the AbstractRun interface poll on the run
        when `get_status` is called, but KubernetesSubmittedRun uses
        Kubernetes watch streams to update the run status. One thread handles
        events from the job object and another thread handles events from the
        rank 0 pod. These threads updated the `_status` attributed of the
        KubernetesSubmittedRun object. When `get_status` is called, the
        `_status` attribute is returned.

        Arguments:
            batch_api: Kubernetes BatchV1Api object.
            core_api: Kubernetes CoreV1Api object.
            name: Name of the job.
            namespace: Kubernetes namespace.
            secret: Kubernetes secret.

        Returns:
            None.
        """
        self.batch_api = batch_api
        self.core_api = core_api
        self.name = name
        self.namespace = namespace
        self._fail_count = 0
        self.secret = secret

    @property
    def id(self) -> str:
        """Return the run id."""
        return self.name

    async def get_logs(self) -> Optional[str]:
        try:
            pods = await self.core_api.list_namespaced_pod(label_selector=f'job-name={self.name}', namespace=self.namespace)
            pod_names = [pi.metadata.name for pi in pods.items]
            if not pod_names:
                wandb.termwarn(f'Found no pods for kubernetes job: {self.name}')
                return None
            logs = await self.core_api.read_namespaced_pod_log(name=pod_names[0], namespace=self.namespace)
            if logs:
                return str(logs)
            else:
                wandb.termwarn(f'No logs for kubernetes pod(s): {pod_names}')
            return None
        except Exception as e:
            wandb.termerror(f'{LOG_PREFIX}Failed to get pod logs: {e}')
            return None

    async def wait(self) -> bool:
        """Wait for the run to finish.

        Returns:
            True if the run finished successfully, False otherwise.
        """
        while True:
            status = await self.get_status()
            wandb.termlog(f'{LOG_PREFIX}Job {self.name} status: {status.state}')
            if status.state in ['finished', 'failed', 'preempted']:
                break
            await asyncio.sleep(5)
        await self._delete_secret()
        return status.state == 'finished'

    async def get_status(self) -> Status:
        status = LaunchKubernetesMonitor.get_status(self.name)
        if status in ['stopped', 'failed', 'finished', 'preempted']:
            await self._delete_secret()
        return status

    async def cancel(self) -> None:
        """Cancel the run."""
        try:
            await self.batch_api.delete_namespaced_job(namespace=self.namespace, name=self.name)
            await self._delete_secret()
        except ApiException as e:
            raise LaunchError(f'Failed to delete Kubernetes Job {self.name} in namespace {self.namespace}: {str(e)}') from e

    async def _delete_secret(self) -> None:
        if not os.environ.get('WANDB_RELEASE_NAME') and self.secret:
            await self.core_api.delete_namespaced_secret(name=self.secret.metadata.name, namespace=self.secret.metadata.namespace)
            self.secret = None