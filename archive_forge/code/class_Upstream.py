import asyncio
import logging
from pprint import pprint
from typing import Dict, Union
import aiohttp
from starlette.requests import Request
import ray
from ray import serve
from ray.serve._private.benchmarks.common import run_throughput_benchmark
from ray.serve.handle import RayServeHandle
@serve.deployment(max_concurrent_queries=1000)
class Upstream:

    def __init__(self, handle: RayServeHandle):
        self._handle = handle
        logging.getLogger('ray.serve').setLevel(logging.WARNING)

    async def __call__(self, req: Request):
        return await self._handle.remote(await req.body())