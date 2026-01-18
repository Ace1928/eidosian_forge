import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeEndpointGroup(object):
    """Fake one or more endpoint group."""

    @staticmethod
    def create_one_endpointgroup(attrs=None):
        """Create a fake endpoint group.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, url, and so on
        """
        attrs = attrs or {}
        endpointgroup_info = {'id': 'endpoint-group-id-' + uuid.uuid4().hex, 'name': 'endpoint-group-name-' + uuid.uuid4().hex, 'filters': {'region': 'region-' + uuid.uuid4().hex, 'service_id': 'service-id-' + uuid.uuid4().hex}, 'description': 'endpoint-group-description-' + uuid.uuid4().hex, 'links': 'links-' + uuid.uuid4().hex}
        endpointgroup_info.update(attrs)
        endpoint = fakes.FakeResource(info=copy.deepcopy(endpointgroup_info), loaded=True)
        return endpoint

    @staticmethod
    def create_one_endpointgroup_filter(attrs=None):
        """Create a fake endpoint project relationship.

        :param Dictionary attrs:
            A dictionary with all attributes of endpointgroup filter
        :return:
            A FakeResource object with project, endpointgroup and so on
        """
        attrs = attrs or {}
        endpointgroup_filter_info = {'project': 'project-id-' + uuid.uuid4().hex, 'endpointgroup': 'endpointgroup-id-' + uuid.uuid4().hex}
        endpointgroup_filter_info.update(attrs)
        endpointgroup_filter = fakes.FakeModel(copy.deepcopy(endpointgroup_filter_info))
        return endpointgroup_filter