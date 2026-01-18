from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def _get_obj_with_vers(self, vers):
    return mock.MagicMock(api_version=api_versions.APIVersion(vers))