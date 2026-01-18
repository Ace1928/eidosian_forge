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
class JobInfoStorageClient:
    """
    Interface to put and get job data from the Internal KV store.
    """
    JOB_DATA_KEY_PREFIX = f'{ray_constants.RAY_INTERNAL_NAMESPACE_PREFIX}job_info_'
    JOB_DATA_KEY = f'{JOB_DATA_KEY_PREFIX}{{job_id}}'

    def __init__(self, gcs_aio_client: GcsAioClient):
        self._gcs_aio_client = gcs_aio_client
        assert _internal_kv_initialized()

    async def put_info(self, job_id: str, job_info: JobInfo, overwrite: bool=True) -> bool:
        """Put job info to the internal kv store.

        Args:
            job_id: The job id.
            job_info: The job info.
            overwrite: Whether to overwrite the existing job info.

        Returns:
            True if a new key is added.
        """
        added_num = await self._gcs_aio_client.internal_kv_put(self.JOB_DATA_KEY.format(job_id=job_id).encode(), json.dumps(job_info.to_json()).encode(), overwrite, namespace=ray_constants.KV_NAMESPACE_JOB)
        return added_num == 1

    async def get_info(self, job_id: str, timeout: int=30) -> Optional[JobInfo]:
        serialized_info = await self._gcs_aio_client.internal_kv_get(self.JOB_DATA_KEY.format(job_id=job_id).encode(), namespace=ray_constants.KV_NAMESPACE_JOB, timeout=timeout)
        if serialized_info is None:
            return None
        else:
            return JobInfo.from_json(json.loads(serialized_info))

    async def delete_info(self, job_id: str, timeout: int=30):
        await self._gcs_aio_client.internal_kv_del(self.JOB_DATA_KEY.format(job_id=job_id).encode(), False, namespace=ray_constants.KV_NAMESPACE_JOB, timeout=timeout)

    async def put_status(self, job_id: str, status: JobStatus, message: Optional[str]=None, driver_exit_code: Optional[int]=None, jobinfo_replace_kwargs: Optional[Dict[str, Any]]=None):
        """Puts or updates job status.  Sets end_time if status is terminal."""
        old_info = await self.get_info(job_id)
        if jobinfo_replace_kwargs is None:
            jobinfo_replace_kwargs = dict()
        jobinfo_replace_kwargs.update(status=status, message=message, driver_exit_code=driver_exit_code)
        if old_info is not None:
            if status != old_info.status and old_info.status.is_terminal():
                assert False, 'Attempted to change job status from a terminal state.'
            new_info = replace(old_info, **jobinfo_replace_kwargs)
        else:
            new_info = JobInfo(entrypoint='Entrypoint not found.', **jobinfo_replace_kwargs)
        if status.is_terminal():
            new_info.end_time = int(time.time() * 1000)
        await self.put_info(job_id, new_info)

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        job_info = await self.get_info(job_id)
        if job_info is None:
            return None
        else:
            return job_info.status

    async def get_all_jobs(self, timeout: int=30) -> Dict[str, JobInfo]:
        raw_job_ids_with_prefixes = await self._gcs_aio_client.internal_kv_keys(self.JOB_DATA_KEY_PREFIX.encode(), namespace=ray_constants.KV_NAMESPACE_JOB, timeout=timeout)
        job_ids_with_prefixes = [job_id.decode() for job_id in raw_job_ids_with_prefixes]
        job_ids = []
        for job_id_with_prefix in job_ids_with_prefixes:
            assert job_id_with_prefix.startswith(self.JOB_DATA_KEY_PREFIX), 'Unexpected format for internal_kv key for Job submission'
            job_ids.append(job_id_with_prefix[len(self.JOB_DATA_KEY_PREFIX):])

        async def get_job_info(job_id: str):
            job_info = await self.get_info(job_id, timeout)
            return (job_id, job_info)
        return {job_id: job_info for job_id, job_info in await asyncio.gather(*[get_job_info(job_id) for job_id in job_ids])}