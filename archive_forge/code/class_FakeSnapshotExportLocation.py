import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeSnapshotExportLocation(object):
    """Fake one or more export locations"""

    @staticmethod
    def create_one_export_location(attrs=None):
        """Create a fake snapshot export location

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        snapshot_export_location_info = {'created_at': 'time-' + uuid.uuid4().hex, 'id': 'id-' + uuid.uuid4().hex, 'is_admin_only': False, 'links': [], 'path': '/path/to/fake/snapshot/snapshot', 'share_snapshot_instance_id': 'instance-id' + uuid.uuid4().hex, 'updated_at': 'time-' + uuid.uuid4().hex}
        snapshot_export_location_info.update(attrs)
        snapshot_export_location = osc_fakes.FakeResource(info=copy.deepcopy(snapshot_export_location_info), loaded=True)
        return snapshot_export_location

    @staticmethod
    def create_export_locations(attrs={}, count=2):
        """Create multiple fake export locations.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share types to be faked
        :return:
            A list of FakeResource objects
        """
        export_locations = []
        for n in range(0, count):
            export_locations.append(FakeSnapshotExportLocation.create_one_export_location(attrs))
        return export_locations