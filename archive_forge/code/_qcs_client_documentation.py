import asyncio
from contextlib import contextmanager
from typing import Iterator
import httpx
from qcs_api_client.client import QCSClientConfiguration, build_sync_client

    Build a QCS client.

    :param client_configuration: Client configuration.
    :param request_timeout: Time limit for requests, in seconds.
    