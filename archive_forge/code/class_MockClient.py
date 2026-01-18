import os
import sys
import time
import unittest
from unittest.mock import Mock, patch
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.utils.py3 import u, httplib, assertRaisesRegex
from libcloud.compute.ssh import BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.compute.base import Node, NodeAuthPassword
from libcloud.test.secrets import RACKSPACE_PARAMS
from libcloud.compute.types import NodeState, LibcloudError, DeploymentError
from libcloud.compute.deployment import (
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rackspace import RackspaceFirstGenNodeDriver as Rackspace
class MockClient(BaseSSHClient):

    def __init__(self, throw_on_timeout=False, *args, **kwargs):
        self.stdout = ''
        self.stderr = ''
        self.exit_status = 0
        self.throw_on_timeout = throw_on_timeout

    def put(self, path, contents, chmod=755, mode='w'):
        return contents

    def putfo(self, path, fo, chmod=755):
        return fo.read()

    def run(self, cmd, timeout=None):
        if self.throw_on_timeout and timeout is not None:
            raise ValueError('timeout')
        return (self.stdout, self.stderr, self.exit_status)

    def delete(self, name):
        return True