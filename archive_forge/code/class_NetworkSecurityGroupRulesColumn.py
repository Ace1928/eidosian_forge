import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class NetworkSecurityGroupRulesColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        return _format_network_security_group_rules(self._value)