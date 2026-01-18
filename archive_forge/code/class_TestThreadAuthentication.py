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
class TestThreadAuthentication(AuthTest):
    """Test authentication running in a thread"""

    def make_auth(self):
        from zmq.auth.thread import ThreadAuthenticator
        return ThreadAuthenticator(self.context)