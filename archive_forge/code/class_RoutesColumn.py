import copy
import json
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class RoutesColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        for route in self._value or []:
            if 'nexthop' in route:
                route['gateway'] = route.pop('nexthop')
        return utils.format_list_of_dicts(self._value)