import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
class TestNodeState(TestCase):

    def test_nodestate_tostring(self):
        self.assertEqual(NodeState.tostring(NodeState.RUNNING), 'RUNNING')

    def test_nodestate_fromstring(self):
        self.assertEqual(NodeState.fromstring('running'), NodeState.RUNNING)