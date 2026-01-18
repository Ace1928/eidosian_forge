import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class AddressesColumn(cliff_columns.FormattableColumn):
    """Generate a formatted string of a server's addresses."""

    def human_readable(self):
        try:
            return utils.format_dict_of_list({k: [i['addr'] for i in v if 'addr' in i] for k, v in self._value.items()})
        except Exception:
            return 'N/A'

    def machine_readable(self):
        return {k: [i['addr'] for i in v if 'addr' in i] for k, v in self._value.items()}