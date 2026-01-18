from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
def _make_rule(rule_type='fake-rule-type', params=None):
    mock_rule = mock.MagicMock()
    mock_rule.rule_type = rule_type
    params = params or {}
    mock_rule.get = params.get
    return mock_rule