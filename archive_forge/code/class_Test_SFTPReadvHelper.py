import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class Test_SFTPReadvHelper(tests.TestCase):

    def checkGetRequests(self, expected_requests, offsets):
        self.requireFeature(features.paramiko)
        helper = _mod_sftp._SFTPReadvHelper(offsets, 'artificial_test', _null_report_activity)
        self.assertEqual(expected_requests, helper._get_requests())

    def test__get_requests(self):
        self.checkGetRequests([(0, 100)], [(0, 20), (30, 50), (20, 10), (80, 20)])
        self.checkGetRequests([(0, 20), (30, 50)], [(10, 10), (30, 20), (0, 10), (50, 30)])
        self.checkGetRequests([(0, 32768), (32768, 32768), (65536, 464)], [(0, 40000), (40000, 100), (40100, 1900), (42000, 24000)])

    def checkRequestAndYield(self, expected, data, offsets):
        self.requireFeature(features.paramiko)
        helper = _mod_sftp._SFTPReadvHelper(offsets, 'artificial_test', _null_report_activity)
        data_f = ReadvFile(data)
        result = list(helper.request_and_yield_offsets(data_f))
        self.assertEqual(expected, result)

    def test_request_and_yield_offsets(self):
        data = b'abcdefghijklmnopqrstuvwxyz'
        self.checkRequestAndYield([(0, b'a'), (5, b'f'), (10, b'klm')], data, [(0, 1), (5, 1), (10, 3)])
        self.checkRequestAndYield([(0, b'a'), (1, b'b'), (10, b'klm')], data, [(0, 1), (1, 1), (10, 3)])
        self.checkRequestAndYield([(0, b'a'), (10, b'k'), (4, b'efg'), (1, b'bcd')], data, [(0, 1), (10, 1), (4, 3), (1, 3)])