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
@pytest.mark.skipif(sys.platform == 'win32' and sys.version_info < (3, 8), reason='flaky event loop cleanup on windows+py<38')
class TestAsyncioAuthentication(AuthTest):
    """Test authentication running in a thread"""

    def make_auth(self):
        from zmq.auth.asyncio import AsyncioAuthenticator
        return AsyncioAuthenticator(self.context)