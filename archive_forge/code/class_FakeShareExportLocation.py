import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareExportLocation(object):
    """Fake one or more export locations"""

    @staticmethod
    def create_one_export_location(attrs=None):
        """Create a fake share export location

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        share_export_location_info = {'created_at': 'time-' + uuid.uuid4().hex, 'fake_path': '/foo/el/path', 'fake_share_instance_id': 'share-instance-id' + uuid.uuid4().hex, 'fake_uuid': 'foo_el_uuid', 'id': 'id-' + uuid.uuid4().hex, 'is_admin_only': False, 'preferred': False, 'updated_at': 'time-' + uuid.uuid4().hex}
        share_export_location_info.update(attrs)
        share_export_location = osc_fakes.FakeResource(info=copy.deepcopy(share_export_location_info), loaded=True)
        return share_export_location

    @staticmethod
    def create_share_export_locations(attrs=None, count=2):
        """Create multiple fake export locations.

        :param Dictionary attrs:
            A dictionary with all attributes

        :param Integer count:
            The number of share export locations to be faked

        :return:
            A list of FakeResource objects
        """
        share_export_locations = []
        for n in range(0, count):
            share_export_locations.append(FakeShareExportLocation.create_one_export_location(attrs))
        return share_export_locations