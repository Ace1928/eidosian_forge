import asyncio
import json
import logging
import time
from random import random
from typing import Callable, Dict
import aiohttp
import numpy as np
import pandas as pd
from grpc import aio
from starlette.requests import Request
import ray
from ray import serve
from ray.serve._private.common import RequestProtocol
from ray.serve.config import gRPCOptions
from ray.serve.generated import serve_pb2, serve_pb2_grpc
from ray.serve.handle import RayServeHandle
@ray.remote
class gRPCClient:

    def __init__(self):
        channel = aio.insecure_channel('localhost:9000')
        self.stub = serve_pb2_grpc.RayServeBenchmarkServiceStub(channel)

    def ready(self):
        return 'ok'

    async def do_queries(self, num, data):
        for _ in range(num):
            await fetch_grpc(self.stub, data)

    async def time_queries(self, num, data):
        stats = []
        for _ in range(num):
            start = time.time()
            await fetch_grpc(self.stub, data)
            end = time.time()
            stats.append(end - start)
        return stats