import copy
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_types
def _get_share_types_manager(self, microversion):
    version = api_versions.APIVersion(microversion)
    mock_microversion = mock.Mock(api_version=version)
    return share_types.ShareTypeManager(api=mock_microversion)