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
@staticmethod
def _filter_match(data, conditions):
    for key, value in conditions.items():
        try:
            if getattr(data, key) != value:
                return False
        except AttributeError:
            continue
    return True