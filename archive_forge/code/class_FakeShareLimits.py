import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareLimits(object):
    """Fake one or more share limits"""

    @staticmethod
    def create_one_share_limit(attrs=None):
        """Create a fake share limit dict

         :param Dictionary attrs:
             A dictionary with all attributes
         :return:
             A FakeLimitsResource object, with share limits.
         """
        attrs = attrs or {}
        share_limits = {'absolute_limit': {'totalShareNetworksUsed': 4}, 'rate_limit': {'regex': '^/shares', 'uri': '/shares', 'verb': 'GET', 'next-available': '2021-09-01T00:00:00Z', 'unit': 'MINUTE', 'value': '3', 'remaining': '1'}}
        share_limits.update(attrs)
        share_limits = osc_fakes.FakeLimitsResource(info=copy.deepcopy(share_limits), loaded=True)
        return share_limits