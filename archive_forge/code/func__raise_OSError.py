import errno
import os
import subprocess
from unittest import mock
import testresources
from testtools import matchers
from keystoneclient.common import cms
from keystoneclient import exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
def _raise_OSError(*args):
    e = OSError()
    e.errno = errno.EPIPE
    raise e