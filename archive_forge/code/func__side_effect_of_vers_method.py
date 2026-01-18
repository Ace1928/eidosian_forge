from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def _side_effect_of_vers_method(self, *args, **kwargs):
    m = mock.MagicMock(start_version=args[1], end_version=args[2])
    m.name = args[0]
    return m