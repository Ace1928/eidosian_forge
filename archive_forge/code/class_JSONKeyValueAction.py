import argparse
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
class JSONKeyValueAction(argparse.Action):
    """A custom action to parse arguments as JSON or key=value pairs

    Ensures that ``dest`` is a dict
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, {})
        current_dest = getattr(namespace, self.dest)
        try:
            current_dest.update(json.loads(values))
        except ValueError as e:
            if '=' in values:
                current_dest.update([values.split('=', 1)])
            else:
                msg = _("Expected '<key>=<value>' or JSON data for option %(option)s, but encountered JSON parsing error: %(error)s") % {'option': option_string, 'error': e}
                raise argparse.ArgumentError(self, msg)