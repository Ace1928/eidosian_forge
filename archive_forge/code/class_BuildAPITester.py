import asyncio
import json
import os
from tempfile import TemporaryDirectory
import pytest
import tornado
class BuildAPITester:
    """Wrapper for build REST API requests"""
    url = 'lab/api/build'

    def __init__(self, labapp, fetch_long):
        self.labapp = labapp
        self.fetch = fetch_long

    async def _req(self, verb, path, body=None):
        return await self.fetch(self.url + path, method=verb, body=body)

    async def getStatus(self):
        return await self._req('GET', '')

    async def build(self):
        return await self._req('POST', '', json.dumps({}))

    async def clear(self):
        return await self._req('DELETE', '')