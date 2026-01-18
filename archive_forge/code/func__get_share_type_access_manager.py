from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import share_type_access
def _get_share_type_access_manager(self, microversion):
    version = api_versions.APIVersion(microversion)
    mock_microversion = mock.Mock(api_version=version)
    return share_type_access.ShareTypeAccessManager(api=mock_microversion)