import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class DeprecationWarningTestsNoOsloLog(DeprecationWarningTests):
    log_prefix = ''

    def setUp(self):
        super(DeprecationWarningTestsNoOsloLog, self).setUp()
        self.useFixture(fixtures.MockPatchObject(cfg, 'oslo_log', None))