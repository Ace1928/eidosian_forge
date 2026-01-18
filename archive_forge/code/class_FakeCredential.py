import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class FakeCredential(object):
    """Fake one or more credential."""

    @staticmethod
    def create_one_credential(attrs=None):
        """Create a fake credential.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, type, and so on
        """
        attrs = attrs or {}
        credential_info = {'id': 'credential-id-' + uuid.uuid4().hex, 'type': 'cert', 'user_id': 'user-id-' + uuid.uuid4().hex, 'blob': 'credential-data-' + uuid.uuid4().hex, 'project_id': 'project-id-' + uuid.uuid4().hex, 'links': 'links-' + uuid.uuid4().hex}
        credential_info.update(attrs)
        credential = fakes.FakeResource(info=copy.deepcopy(credential_info), loaded=True)
        return credential

    @staticmethod
    def create_credentials(attrs=None, count=2):
        """Create multiple fake credentials.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of credentials to fake
        :return:
            A list of FakeResource objects faking the credentials
        """
        credentials = []
        for i in range(0, count):
            credential = FakeCredential.create_one_credential(attrs)
            credentials.append(credential)
        return credentials

    @staticmethod
    def get_credentials(credentials=None, count=2):
        """Get an iterable MagicMock object with a list of faked credentials.

        If credentials list is provided, then initialize the Mock object with
        the list. Otherwise create one.

        :param List credentials:
            A list of FakeResource objects faking credentials
        :param Integer count:
            The number of credentials to be faked
        :return:
            An iterable Mock object with side_effect set to a list of faked
            credentials
        """
        if credentials is None:
            credentials = FakeCredential.create_credentials(count)
        return mock.Mock(side_effect=credentials)