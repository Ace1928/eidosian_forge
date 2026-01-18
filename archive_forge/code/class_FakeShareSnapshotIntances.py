import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareSnapshotIntances(object):
    """Fake a share snapshot instance"""

    @staticmethod
    def create_one_snapshot_instance(attrs=None, methods=None):
        """Create a fake share snapshot instance

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        methods = methods or {}
        share_snapshot_instance = {'id': 'snapshot-instance-id-' + uuid.uuid4().hex, 'snapshot_id': 'snapshot-id-' + uuid.uuid4().hex, 'status': None, 'created_at': datetime.datetime.now().isoformat(), 'updated_at': datetime.datetime.now().isoformat(), 'share_id': 'share-id-' + uuid.uuid4().hex, 'share_instance_id': 'share-instance-id-' + uuid.uuid4().hex, 'progress': None, 'provider_location': None}
        share_snapshot_instance.update(attrs)
        share_snapshot_instance = osc_fakes.FakeResource(info=copy.deepcopy(share_snapshot_instance), methods=methods, loaded=True)
        return share_snapshot_instance

    @staticmethod
    def create_share_snapshot_instances(attrs=None, count=2):
        """Create multiple fake snapshot instances.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share snapshot instances to be faked
        :return:
            A list of FakeResource objects
        """
        share_snapshot_instances = []
        for n in range(0, count):
            share_snapshot_instances.append(FakeShareSnapshot.create_one_snapshot(attrs))
        return share_snapshot_instances