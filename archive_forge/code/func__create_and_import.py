import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
def _create_and_import(self, stores=[], data_iter=None, expected_code=202, visibility=None, extra={}):
    """Create an image, stage data, and import into the given stores.

        :returns: image_id
        """
    image_id = self._create_and_stage(data_iter=data_iter, visibility=visibility, extra=extra)
    resp = self._import_direct(image_id, stores)
    self.assertEqual(expected_code, resp.status_code)
    if expected_code >= 400:
        return image_id
    image = self._wait_for_import(image_id)
    self.assertEqual('active', image['status'])
    return image_id