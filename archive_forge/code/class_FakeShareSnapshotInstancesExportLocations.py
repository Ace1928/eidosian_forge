import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareSnapshotInstancesExportLocations(object):
    """Fake a share snapshot instance Export Locations"""

    @staticmethod
    def create_one_snapshot_instance(attrs=None, methods=None):
        """Create a fake share snapshot instance export locations

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        methods = methods or {}
        share_snapshot_instance_export_location = {'id': 'snapshot-instance-export-location-id-' + uuid.uuid4().hex, 'is_admin_only': False, 'path': '0.0.0.0/:fake-share-instance-export-location-id'}
        share_snapshot_instance_export_location.update(attrs)
        share_snapshot_instance_export_location = osc_fakes.FakeResource(info=copy.deepcopy(share_snapshot_instance_export_location), methods=methods, loaded=True)
        return share_snapshot_instance_export_location

    @staticmethod
    def create_share_snapshot_instances(attrs=None, count=2):
        """Create multiple fake snapshot instances.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share snapshot instance locations to be faked
        :return:
            A list of FakeResource objects
        """
        share_snapshot_instances = []
        for n in range(0, count):
            share_snapshot_instances.append(FakeShareSnapshot.create_one_snapshot(attrs))
        return share_snapshot_instances