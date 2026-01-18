import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
import pytest
import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy
class CredentialsProvider:

    def __init__(self):
        if key == 'ok':
            self.client = client_public
        else:
            self.client = server_public

    def callback(self, domain, key):
        if key == self.client:
            return True
        else:
            return False

    async def async_callback(self, domain, key):
        await asyncio.sleep(0.1)
        if key == self.client:
            return True
        else:
            return False