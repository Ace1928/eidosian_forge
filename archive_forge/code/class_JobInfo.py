import asyncio
import json
import time
from dataclasses import dataclass, replace, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.packaging import parse_uri
from ray.experimental.internal_kv import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
@dataclass
class JobInfo:
    """A class for recording information associated with a job and its execution.

    Please keep this in sync with the JobsAPIInfo proto in src/ray/protobuf/gcs.proto.
    """
    status: JobStatus
    entrypoint: str
    message: Optional[str] = None
    error_type: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    runtime_env: Optional[Dict[str, Any]] = None
    entrypoint_num_cpus: Optional[Union[int, float]] = None
    entrypoint_num_gpus: Optional[Union[int, float]] = None
    entrypoint_memory: Optional[int] = None
    entrypoint_resources: Optional[Dict[str, float]] = None
    driver_agent_http_address: Optional[str] = None
    driver_node_id: Optional[str] = None
    driver_exit_code: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = JobStatus(self.status)
        if self.message is None:
            if self.status == JobStatus.PENDING:
                self.message = 'Job has not started yet.'
                if any([self.entrypoint_num_cpus is not None and self.entrypoint_num_cpus > 0, self.entrypoint_num_gpus is not None and self.entrypoint_num_gpus > 0, self.entrypoint_memory is not None and self.entrypoint_memory > 0, self.entrypoint_resources not in [None, {}]]):
                    self.message += ' It may be waiting for resources (CPUs, GPUs, memory, custom resources) to become available.'
                if self.runtime_env not in [None, {}]:
                    self.message += ' It may be waiting for the runtime environment to be set up.'
            elif self.status == JobStatus.RUNNING:
                self.message = 'Job is currently running.'
            elif self.status == JobStatus.STOPPED:
                self.message = 'Job was intentionally stopped.'
            elif self.status == JobStatus.SUCCEEDED:
                self.message = 'Job finished successfully.'
            elif self.status == JobStatus.FAILED:
                self.message = 'Job failed.'

    def to_json(self) -> Dict[str, Any]:
        """Convert this object to a JSON-serializable dictionary.

        Note that the runtime_env field is converted to a JSON-serialized string
        and the field is renamed to runtime_env_json.

        Returns:
            A JSON-serializable dictionary representing the JobInfo object.
        """
        json_dict = asdict(self)
        json_dict['status'] = str(json_dict['status'])
        if 'runtime_env' in json_dict:
            if json_dict['runtime_env'] is not None:
                json_dict['runtime_env_json'] = json.dumps(json_dict['runtime_env'])
            del json_dict['runtime_env']
        json.dumps(json_dict)
        return json_dict

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> None:
        """Initialize this object from a JSON dictionary.

        Note that the runtime_env_json field is converted to a dictionary and
        the field is renamed to runtime_env.

        Args:
            json_dict: A JSON dictionary to use to initialize the JobInfo object.
        """
        json_dict['status'] = JobStatus(json_dict['status'])
        if 'runtime_env_json' in json_dict:
            if json_dict['runtime_env_json'] is not None:
                json_dict['runtime_env'] = json.loads(json_dict['runtime_env_json'])
            del json_dict['runtime_env_json']
        return cls(**json_dict)