import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _get_encryption_info(self):
    type_id = self._value
    return self._encryption_data.get(type_id)