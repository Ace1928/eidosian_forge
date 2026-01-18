import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareAvailabilityZones(object):
    """Fake one or more availability zones"""

    @staticmethod
    def create_one_availability_zone(attrs=None):
        """Create a fake share availability zone

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        availability_zone = {'id': 'id-' + uuid.uuid4().hex, 'name': 'name-' + uuid.uuid4().hex, 'created_at': 'time-' + uuid.uuid4().hex, 'updated_at': 'time-' + uuid.uuid4().hex}
        availability_zone.update(attrs)
        availability_zone = osc_fakes.FakeResource(info=copy.deepcopy(availability_zone), loaded=True)
        return availability_zone

    @staticmethod
    def create_share_availability_zones(attrs=None, count=2):
        """Create multiple availability zones.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of availability zones to be faked
        :return:
            A list of FakeResource objects
        """
        availability_zones = []
        for n in range(0, count):
            availability_zones.append(FakeShareAvailabilityZones.create_one_availability_zone(attrs))
        return availability_zones