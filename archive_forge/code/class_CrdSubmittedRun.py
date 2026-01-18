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
class CrdSubmittedRun(AbstractRun):
    """Run submitted to a CRD backend, e.g. Volcano."""

    def __init__(self, group: str, version: str, plural: str, name: str, namespace: str, core_api: CoreV1Api, custom_api: CustomObjectsApi) -> None:
        """Create a run object for tracking the progress of a CRD.

        Arguments:
            group: The API group of the CRD.
            version: The API version of the CRD.
            plural: The plural name of the CRD.
            name: The name of the CRD instance.
            namespace: The namespace of the CRD instance.
            core_api: The Kubernetes core API client.
            custom_api: The Kubernetes custom object API client.

        Raises:
            LaunchError: If the CRD instance does not exist.
        """
        self.group = group
        self.version = version
        self.plural = plural
        self.name = name
        self.namespace = namespace
        self.core_api = core_api
        self.custom_api = custom_api
        self._fail_count = 0

    @property
    def id(self) -> str:
        """Get the name of the custom object."""
        return self.name

    async def get_logs(self) -> Optional[str]:
        """Get logs for custom object."""
        logs: Dict[str, Optional[str]] = {}
        try:
            pods = await self.core_api.list_namespaced_pod(label_selector=f'wandb/run-id={self.name}', namespace=self.namespace)
            pod_names = [pi.metadata.name for pi in pods.items]
            for pod_name in pod_names:
                logs[pod_name] = await self.core_api.read_namespaced_pod_log(name=pod_name, namespace=self.namespace)
        except ApiException as e:
            wandb.termwarn(f'Failed to get logs for {self.name}: {str(e)}')
            return None
        if not logs:
            return None
        logs_as_array = [f'Pod {pod_name}:\n{log}' for pod_name, log in logs.items()]
        return '\n'.join(logs_as_array)

    async def get_status(self) -> Status:
        """Get status of custom object."""
        return LaunchKubernetesMonitor.get_status(self.name)

    async def cancel(self) -> None:
        """Cancel the custom object."""
        try:
            await self.custom_api.delete_namespaced_custom_object(group=self.group, version=self.version, namespace=self.namespace, plural=self.plural, name=self.name)
        except ApiException as e:
            raise LaunchError(f'Failed to delete CRD {self.name} in namespace {self.namespace}: {str(e)}') from e

    async def wait(self) -> bool:
        """Wait for this custom object to finish running."""
        while True:
            status = await self.get_status()
            wandb.termlog(f'{LOG_PREFIX}Job {self.name} status: {status}')
            if status.state in ['finished', 'failed', 'preempted']:
                return status.state == 'finished'
            await asyncio.sleep(5)