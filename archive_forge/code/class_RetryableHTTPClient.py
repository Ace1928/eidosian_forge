from __future__ import annotations
import asyncio
import json
import logging
import os
import typing as ty
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from socket import gaierror
from jupyter_events import EventLogger
from tornado import web
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPResponse
from traitlets import (
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
class RetryableHTTPClient:
    """
    Inspired by urllib.util.Retry (https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html),
    this class is initialized with desired retry characteristics, uses a recursive method `fetch()` against an instance
    of `AsyncHTTPClient` which tracks the current retry count across applicable request retries.
    """
    MAX_RETRIES_DEFAULT = 2
    MAX_RETRIES_CAP = 10
    max_retries: int = int(os.getenv('JUPYTER_GATEWAY_MAX_REQUEST_RETRIES', MAX_RETRIES_DEFAULT))
    max_retries = max(0, min(max_retries, MAX_RETRIES_CAP))
    retried_methods: set[str] = {'GET', 'DELETE'}
    retried_errors: set[int] = {502, 503, 504, 599}
    retried_exceptions: set[type] = {ConnectionError}
    backoff_factor: float = 0.1

    def __init__(self):
        """Initialize the retryable http client."""
        self.retry_count: int = 0
        self.client: AsyncHTTPClient = AsyncHTTPClient()

    async def fetch(self, endpoint: str, **kwargs: ty.Any) -> HTTPResponse:
        """
        Retryable AsyncHTTPClient.fetch() method.  When the request fails, this method will
        recurse up to max_retries times if the condition deserves a retry.
        """
        self.retry_count = 0
        return await self._fetch(endpoint, **kwargs)

    async def _fetch(self, endpoint: str, **kwargs: ty.Any) -> HTTPResponse:
        """
        Performs the fetch against the contained AsyncHTTPClient instance and determines
        if retry is necessary on any exceptions.  If so, retry is performed recursively.
        """
        try:
            response: HTTPResponse = await self.client.fetch(endpoint, **kwargs)
        except Exception as e:
            is_retryable: bool = await self._is_retryable(kwargs['method'], e)
            if not is_retryable:
                raise e
            logging.getLogger('ServerApp').info(f"Attempting retry ({self.retry_count}) against endpoint '{endpoint}'.  Retried error: '{e!r}'")
            response = await self._fetch(endpoint, **kwargs)
        return response

    async def _is_retryable(self, method: str, exception: Exception) -> bool:
        """Determines if the given exception is retryable based on object's configuration."""
        if method not in self.retried_methods:
            return False
        if self.retry_count == self.max_retries:
            return False
        if isinstance(exception, HTTPClientError):
            hce: HTTPClientError = exception
            if hce.code not in self.retried_errors:
                return False
        elif not any((isinstance(exception, error) for error in self.retried_exceptions)):
            return False
        await asyncio.sleep(self.backoff_factor * 2 ** self.retry_count)
        self.retry_count += 1
        return True