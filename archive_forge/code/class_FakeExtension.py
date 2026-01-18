import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeExtension(object):
    """Fake one or more extension."""

    @staticmethod
    def create_one_extension(attrs=None):
        """Create a fake extension.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object with name, namespace, etc.
        """
        attrs = attrs or {}
        extension_info = {'name': 'name-' + uuid.uuid4().hex, 'namespace': 'http://docs.openstack.org/identity/api/ext/OS-KSCRUD/v1.0', 'description': 'description-' + uuid.uuid4().hex, 'updated': '2013-07-07T12:00:0-00:00', 'alias': 'OS-KSCRUD', 'links': '[{"href":"https://github.com/openstack/identity-api", "type": "text/html", "rel": "describedby"}]'}
        extension_info.update(attrs)
        extension = fakes.FakeResource(info=copy.deepcopy(extension_info), loaded=True)
        return extension