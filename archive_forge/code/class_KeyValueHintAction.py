import argparse
import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class KeyValueHintAction(argparse.Action):
    """Uses KeyValueAction or KeyValueAppendAction based on the given key"""
    APPEND_KEYS = ('same_host', 'different_host')

    def __init__(self, *args, **kwargs):
        self._key_value_action = parseractions.KeyValueAction(*args, **kwargs)
        self._key_value_append_action = parseractions.KeyValueAppendAction(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values.startswith(self.APPEND_KEYS):
            self._key_value_append_action(parser, namespace, values, option_string=option_string)
        else:
            self._key_value_action(parser, namespace, values, option_string=option_string)