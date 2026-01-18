import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareGroupType(object):
    """Fake one or more share group types"""

    @staticmethod
    def create_one_share_group_type(attrs=None, methods=None):
        """Create a fake share group type

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
        attrs = attrs or {}
        methods = methods or {}
        share_group_type_info = {'is_public': True, 'group_specs': {'snapshot_support': True}, 'share_types': ['share-types-id-' + uuid.uuid4().hex], 'id': 'share-group-type-id-' + uuid.uuid4().hex, 'name': 'share-group-type-name-' + uuid.uuid4().hex, 'is_default': False}
        share_group_type_info.update(attrs)
        share_group_type = osc_fakes.FakeResource(info=copy.deepcopy(share_group_type_info), methods=methods, loaded=True)
        return share_group_type

    @staticmethod
    def create_share_group_types(attrs=None, count=2):
        """Create multiple fake share group types.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share group types to be faked
        :return:
            A list of FakeResource objects
        """
        share_group_types = []
        for n in range(0, count):
            share_group_types.append(FakeShareGroupType.create_one_share_group_type(attrs))
        return share_group_types

    @staticmethod
    def get_share_group_types(share_group_types=None, count=2):
        """Get an iterable MagicMock object with a list of faked group types.

        If types list is provided, then initialize the Mock object with the
        list. Otherwise create one.

        :param List types:
            A list of FakeResource objects faking types
        :param Integer count:
            The number of group types to be faked
        :return
            An iterable Mock object with side_effect set to a list of faked
            group types
        """
        if share_group_types is None:
            share_group_types = FakeShareGroupType.share_group_types(count)
        return mock.Mock(side_effect=share_group_types)