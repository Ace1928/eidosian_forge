import io
import json
import os
from unittest import mock
import urllib
import glance_store
from oslo_concurrency import processutils as putils
from oslo_config import cfg
from taskflow import task
from taskflow.types import failure
import glance.async_.flows.base_import as import_flow
from glance.async_ import taskflow_executor
from glance.async_ import utils as async_utils
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import utils
from glance import context
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def _assert_qemu_process_limits(self, exec_mock):
    kw_args = exec_mock.call_args[1]
    self.assertIn('prlimit', kw_args)
    self.assertEqual(async_utils.QEMU_IMG_PROC_LIMITS, kw_args.get('prlimit'))