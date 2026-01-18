import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import kubernetes_asyncio  # type: ignore # noqa: F401
import urllib3
from kubernetes_asyncio import watch
from kubernetes_asyncio.client import (  # type: ignore  # noqa: F401
import wandb
from wandb.sdk.launch.agent import LaunchAgent
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.runner.abstract import State, Status
from wandb.sdk.launch.utils import get_kube_context_and_api_client
class LaunchKubernetesMonitor:
    """Monitors kubernetes resources managed by the launch agent.

    Note: this class is forced to be a singleton in order to prevent multiple
    threads from being created that monitor the same kubernetes resources.
    """
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> 'LaunchKubernetesMonitor':
        """Create a new instance of the LaunchKubernetesMonitor.

        This method ensures that only one instance of the LaunchKubernetesMonitor
        is created. This is done to prevent multiple threads from being created
        that monitor the same kubernetes resources.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, core_api: CoreV1Api, batch_api: BatchV1Api, custom_api: CustomObjectsApi, label_selector: str):
        """Initialize the LaunchKubernetesMonitor."""
        self._core_api: CoreV1Api = core_api
        self._batch_api: BatchV1Api = batch_api
        self._custom_api: CustomObjectsApi = custom_api
        self._label_selector: str = label_selector
        self._monitor_tasks: Dict[Tuple[str, Union[str, CustomResource]], asyncio.Task] = dict()
        self._job_states: Dict[str, Status] = dict()

    @classmethod
    async def ensure_initialized(cls) -> None:
        """Initialize the LaunchKubernetesMonitor."""
        if cls._instance is None:
            _, api_client = await get_kube_context_and_api_client(kubernetes_asyncio, {})
            core_api = CoreV1Api(api_client)
            batch_api = BatchV1Api(api_client)
            custom_api = CustomObjectsApi(api_client)
            label_selector = f'{WANDB_K8S_LABEL_MONITOR}=true'
            if LaunchAgent.initialized():
                label_selector += f',{WANDB_K8S_LABEL_AGENT}={LaunchAgent.name()}'
            cls(core_api=core_api, batch_api=batch_api, custom_api=custom_api, label_selector=label_selector)

    @classmethod
    def monitor_namespace(cls, namespace: str, custom_resource: Optional[CustomResource]=None) -> None:
        """Start monitoring a namespaces for resources."""
        if cls._instance is None:
            raise LaunchError('LaunchKubernetesMonitor not initialized, cannot monitor namespace.')
        cls._instance.__monitor_namespace(namespace, custom_resource=custom_resource)

    @classmethod
    def get_status(cls, job_name: str) -> Status:
        """Get the status of a job."""
        if cls._instance is None:
            raise LaunchError('LaunchKubernetesMonitor not initialized, cannot get status.')
        return cls._instance.__get_status(job_name)

    @classmethod
    def status_count(cls) -> Dict[State, int]:
        """Get a dictionary mapping statuses to the # monitored jobs with each status."""
        if cls._instance is None:
            raise ValueError('LaunchKubernetesMonitor not initialized, cannot get status counts.')
        return cls._instance.__status_count()

    def __monitor_namespace(self, namespace: str, custom_resource: Optional[CustomResource]=None) -> None:
        """Start monitoring a namespaces for resources."""
        if (namespace, Resources.PODS) not in self._monitor_tasks:
            self._monitor_tasks[namespace, Resources.PODS] = create_named_task(f'monitor_pods_{namespace}', self._monitor_pods, namespace)
        if custom_resource is not None:
            if (namespace, custom_resource) not in self._monitor_tasks:
                self._monitor_tasks[namespace, custom_resource] = create_named_task(f'monitor_{custom_resource}_{namespace}', self._monitor_crd, namespace, custom_resource=custom_resource)
        elif (namespace, Resources.JOBS) not in self._monitor_tasks:
            self._monitor_tasks[namespace, Resources.JOBS] = create_named_task(f'monitor_jobs_{namespace}', self._monitor_jobs, namespace)

    def __get_status(self, job_name: str) -> Status:
        """Get the status of a job."""
        if job_name not in self._job_states:
            return Status('unknown')
        state = self._job_states[job_name]
        return state

    def __status_count(self) -> Dict[State, int]:
        """Get a dictionary mapping statuses to the # monitored jobs with each status."""
        counts = dict()
        for _, status in self._job_states.items():
            state = status.state
            if state not in counts:
                counts[state] = 1
            else:
                counts[state] += 1
        return counts

    def _set_status(self, job_name: str, status: Status) -> None:
        """Set the status of the run."""
        if self._job_states.get(job_name) != status:
            self._job_states[job_name] = status

    async def _monitor_pods(self, namespace: str) -> None:
        """Monitor a namespace for changes."""
        watcher = SafeWatch(watch.Watch())
        async for event in watcher.stream(self._core_api.list_namespaced_pod, namespace=namespace, label_selector=self._label_selector):
            obj = event.get('object')
            job_name = obj.metadata.labels.get('job-name')
            if job_name is None or not hasattr(obj, 'status'):
                continue
            if self.__get_status(job_name) in ['finished', 'failed']:
                continue
            if obj.status.phase == 'Running' or _is_container_creating(obj.status):
                self._set_status(job_name, Status('running'))
            elif _is_preempted(obj.status):
                self._set_status(job_name, Status('preempted'))

    async def _monitor_jobs(self, namespace: str) -> None:
        """Monitor a namespace for changes."""
        watcher = SafeWatch(watch.Watch())
        async for event in watcher.stream(self._batch_api.list_namespaced_job, namespace=namespace, label_selector=self._label_selector):
            obj = event.get('object')
            job_name = obj.metadata.name
            if obj.status.succeeded == 1:
                self._set_status(job_name, Status('finished'))
            elif obj.status.failed is not None and obj.status.failed >= 1:
                self._set_status(job_name, Status('failed'))
            if event.get('type') == 'DELETED':
                if self._job_states.get(job_name) != Status('finished'):
                    self._set_status(job_name, Status('failed'))

    async def _monitor_crd(self, namespace: str, custom_resource: CustomResource) -> None:
        """Monitor a namespace for changes."""
        watcher = SafeWatch(watch.Watch())
        async for event in watcher.stream(self._custom_api.list_namespaced_custom_object, namespace=namespace, plural=custom_resource.plural, group=custom_resource.group, version=custom_resource.version, label_selector=self._label_selector):
            object = event.get('object')
            name = object.get('metadata', dict()).get('name')
            status = object.get('status')
            state = None
            if status is None:
                continue
            replicated_jobs_status = status.get('ReplicatedJobsStatus')
            if isinstance(replicated_jobs_status, dict):
                state = _state_from_replicated_status(replicated_jobs_status)
            state_dict = status.get('state')
            if isinstance(state_dict, dict):
                phase = state_dict.get('phase')
                if phase:
                    state = CRD_STATE_DICT.get(phase.lower())
            else:
                conditions = status.get('conditions')
                if isinstance(conditions, list):
                    state = _state_from_conditions(conditions)
                else:
                    _logger.warning(f'Unexpected conditions type {type(conditions)} for CRD watcher in {namespace}')
            if state is None:
                continue
            status = Status(state)
            self._set_status(name, status)