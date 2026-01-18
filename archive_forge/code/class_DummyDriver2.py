import sys
from unittest.mock import Mock
from libcloud.test import unittest
from libcloud.common.base import BaseDriver
class DummyDriver2(BaseDriver):

    def _ex_connection_class_kwargs(self):
        result = {}
        result['timeout'] = 13
        return result