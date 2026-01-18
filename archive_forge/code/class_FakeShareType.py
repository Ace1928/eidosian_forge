import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareType(object):
    """Fake one or more share types"""

    @staticmethod
    def create_one_sharetype(attrs=None, methods=None):
        """Create a fake share type

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        methods = methods or {}
        share_type_info = {'required_extra_specs': {'driver_handles_share_servers': True}, 'share_type_access:is_public': True, 'extra_specs': {'replication_type': 'readable', 'driver_handles_share_servers': True, 'mount_snapshot_support': False, 'revert_to_snapshot_support': False, 'create_share_from_snapshot_support': True, 'snapshot_support': True}, 'id': 'share-type-id-' + uuid.uuid4().hex, 'name': 'share-type-name-' + uuid.uuid4().hex, 'is_default': False, 'description': 'share-type-description-' + uuid.uuid4().hex}
        share_type_info.update(attrs)
        share_type = osc_fakes.FakeResource(info=copy.deepcopy(share_type_info), methods=methods, loaded=True)
        return share_type

    @staticmethod
    def create_share_types(attrs=None, count=2):
        """Create multiple fake share types.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share types to be faked
        :return:
            A list of FakeResource objects
        """
        share_types = []
        for n in range(0, count):
            share_types.append(FakeShareType.create_one_sharetype(attrs))
        return share_types

    @staticmethod
    def get_share_types(share_types=None, count=2):
        """Get an iterable MagicMock object with a list of faked types.

        If types list is provided, then initialize the Mock object with the
        list. Otherwise create one.

        :param List types:
            A list of FakeResource objects faking types
        :param Integer count:
            The number of types to be faked
        :return
            An iterable Mock object with side_effect set to a list of faked
            types
        """
        if share_types is None:
            share_types = FakeShareType.create_share_types(count)
        return mock.Mock(side_effect=share_types)