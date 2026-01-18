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
class FakeLogger:

    def __init__(self, test_case, expected_lvl):
        self.test_case = test_case
        self.expected_lvl = expected_lvl
        self.logged = []

    def log(self, lvl, fmt, *args):
        self.test_case.assertEqual(lvl, self.expected_lvl)
        self.logged.append(fmt % args)