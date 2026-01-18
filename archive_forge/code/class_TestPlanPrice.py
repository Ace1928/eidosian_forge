import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
class TestPlanPrice(unittest.TestCase):

    def setUp(self):
        prices = [{'name': 'uk-lon1', 'server_plan_1xCPU-1GB': {'amount': 1, 'price': 1.488}}, {'name': 'fi-hel1', 'server_plan_1xCPU-1GB': {'amount': 1, 'price': 1.588}}]
        self.pp = PlanPrice(prices)

    def test_zone_prices(self):
        location = NodeLocation(id='fi-hel1', name='Helsinki #1', country='FI', driver=None)
        self.assertEqual(self.pp.get_price('1xCPU-1GB', location), 1.588)

    def test_plan_not_found_in_zone(self):
        location = NodeLocation(id='no_such_location', name='', country='', driver=None)
        self.assertIsNone(self.pp.get_price('1xCPU-1GB', location))

    def test_no_location_given(self):
        self.assertIsNone(self.pp.get_price('1xCPU-1GB'))