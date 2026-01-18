import sys
from unittest.mock import Mock
from libcloud.test import unittest
from libcloud.common.base import BaseDriver
class BaseDriverTestCase(unittest.TestCase):

    def test_timeout_argument_propagation_and_preservation(self):

        class DummyDriver1(BaseDriver):
            pass
        DummyDriver1.connectionCls = Mock()
        DummyDriver1(key='foo')
        call_kwargs = DummyDriver1.connectionCls.call_args[1]
        self.assertIsNone(call_kwargs['timeout'])
        self.assertIsNone(call_kwargs['retry_delay'])

        class DummyDriver1(BaseDriver):
            pass
        DummyDriver1.connectionCls = Mock()
        DummyDriver1(key='foo', timeout=12)
        call_kwargs = DummyDriver1.connectionCls.call_args[1]
        self.assertEqual(call_kwargs['timeout'], 12)
        self.assertIsNone(call_kwargs['retry_delay'])

        class DummyDriver2(BaseDriver):

            def _ex_connection_class_kwargs(self):
                result = {}
                result['timeout'] = 13
                return result
        DummyDriver2.connectionCls = Mock()
        DummyDriver2(key='foo')
        call_kwargs = DummyDriver2.connectionCls.call_args[1]
        DummyDriver2.connectionCls = Mock()
        DummyDriver2(key='foo', timeout=14, retry_delay=10)
        call_kwargs = DummyDriver2.connectionCls.call_args[1]
        self.assertEqual(call_kwargs['timeout'], 14)
        self.assertEqual(call_kwargs['retry_delay'], 10)