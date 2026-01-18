import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class ResetStateOnErrorTest(common.HeatTestCase):

    class DummyStack(object):
        COMPLETE, IN_PROGRESS, FAILED = range(3)
        action = 'something'
        status = COMPLETE

        def __init__(self):
            self.mark_failed = mock.MagicMock()
            self.convergence = False

        @stack.reset_state_on_error
        def raise_exception(self):
            self.status = self.IN_PROGRESS
            raise ValueError('oops')

        @stack.reset_state_on_error
        def raise_exit_exception(self):
            self.status = self.IN_PROGRESS
            raise BaseException('bye')

        @stack.reset_state_on_error
        def succeed(self):
            return 'Hello world'

        @stack.reset_state_on_error
        def fail(self):
            self.status = self.FAILED
            return 'Hello world'

    def test_success(self):
        dummy = self.DummyStack()
        self.assertEqual('Hello world', dummy.succeed())
        self.assertFalse(dummy.mark_failed.called)

    def test_failure(self):
        dummy = self.DummyStack()
        self.assertEqual('Hello world', dummy.fail())
        self.assertFalse(dummy.mark_failed.called)

    def test_reset_state_exception(self):
        dummy = self.DummyStack()
        exc = self.assertRaises(ValueError, dummy.raise_exception)
        self.assertIn('oops', str(exc))
        self.assertTrue(dummy.mark_failed.called)

    def test_reset_state_exit_exception(self):
        dummy = self.DummyStack()
        exc = self.assertRaises(BaseException, dummy.raise_exit_exception)
        self.assertIn('bye', str(exc))
        self.assertTrue(dummy.mark_failed.called)