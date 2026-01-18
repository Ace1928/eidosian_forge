import asyncio
import dataclasses
import json
import logging
import traceback
from random import sample
from typing import Iterator, Optional
import aiohttp.web
from aiohttp.web import Request, Response
from aiohttp.client import ClientResponse
import ray
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.consts as dashboard_consts
from ray.dashboard.datacenter import DataOrganizer
import ray.dashboard.utils as dashboard_utils
from ray._private.runtime_env.packaging import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.pydantic_models import (
from ray.dashboard.modules.job.utils import (
from ray.dashboard.modules.version import (
class JobAgentSubmissionClient:
    """A local client for submitting and interacting with jobs on a specific node
    in the remote cluster.
    Submits requests over HTTP to the job agent on the specific node using the REST API.
    """

    def __init__(self, dashboard_agent_address: str):
        self._agent_address = dashboard_agent_address
        self._session = aiohttp.ClientSession()

    async def _raise_error(self, resp: ClientResponse):
        status = resp.status
        error_text = await resp.text()
        raise RuntimeError(f'Request failed with status code {status}: {error_text}.')

    async def submit_job_internal(self, req: JobSubmitRequest) -> JobSubmitResponse:
        logger.debug(f'Submitting job with submission_id={req.submission_id}.')
        async with self._session.post(f'{self._agent_address}/api/job_agent/jobs/', json=dataclasses.asdict(req)) as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobSubmitResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def stop_job_internal(self, job_id: str) -> JobStopResponse:
        logger.debug(f'Stopping job with job_id={job_id}.')
        async with self._session.post(f'{self._agent_address}/api/job_agent/jobs/{job_id}/stop') as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobStopResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def delete_job_internal(self, job_id: str) -> JobDeleteResponse:
        logger.debug(f'Deleting job with job_id={job_id}.')
        async with self._session.delete(f'{self._agent_address}/api/job_agent/jobs/{job_id}') as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobDeleteResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def get_job_logs_internal(self, job_id: str) -> JobLogsResponse:
        async with self._session.get(f'{self._agent_address}/api/job_agent/jobs/{job_id}/logs') as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobLogsResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def tail_job_logs(self, job_id: str) -> Iterator[str]:
        """Get an iterator that follows the logs of a job."""
        ws = await self._session.ws_connect(f'{self._agent_address}/api/job_agent/jobs/{job_id}/logs/tail')
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                pass

    async def close(self, ignore_error=True):
        try:
            await self._session.close()
        except Exception:
            if not ignore_error:
                raise