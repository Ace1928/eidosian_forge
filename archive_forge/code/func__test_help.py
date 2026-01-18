import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
def _test_help(self, command, required=None):
    if required is None:
        required = ['.*?^usage: ', '.*?^\\s+set-password\\s+Change the admin password', '.*?^See "nova help COMMAND" for help on a specific command']
    stdout, stderr = self.shell(command)
    for r in required:
        self.assertThat(stdout + stderr, matchers.MatchesRegex(r, re.DOTALL | re.MULTILINE))