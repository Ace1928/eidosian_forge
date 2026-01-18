from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import availability_zones
def _get_resource_path(self, microversion):
    if api_versions.APIVersion(microversion) > api_versions.APIVersion('2.6'):
        return availability_zones.RESOURCE_PATH
    return availability_zones.RESOURCE_PATH_LEGACY