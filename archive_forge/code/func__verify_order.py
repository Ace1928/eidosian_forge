import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def _verify_order(self, test_graph, test_list):
    for k, v in test_graph.items():
        for dep in v:
            self.assertTrue(test_list.index(k) < test_list.index(dep))