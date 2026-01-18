import json
import os
import ssl
import sys
import warnings
import logging
import random
import testtools
import unittest
from unittest import mock
from os_ken.base import app_manager  # To suppress cyclic import
from os_ken.controller import controller
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_0_parser
def _test_ports_accessibility(self, ofproto_parser, msgs_len):
    with mock.patch('os_ken.controller.controller.Datapath.set_state'):
        with warnings.catch_warnings(record=True) as msgs:
            warnings.simplefilter('always')
            sock_mock = mock.Mock()
            addr_mock = mock.Mock()
            dp = controller.Datapath(sock_mock, addr_mock)
            dp.ofproto_parser = ofproto_parser
            dp.ports = {}
            port_mock = mock.Mock()
            dp.ports[0] = port_mock
            del dp.ports[0]
            self.assertEqual(len(msgs), msgs_len)
            for msg in msgs:
                self.assertTrue(issubclass(msg.category, UserWarning))