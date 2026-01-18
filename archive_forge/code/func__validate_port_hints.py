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
def _validate_port_hints(hints):
    if hints not in ({}, {'ovs-tx-steering': 'thread'}, {'ovs-tx-steering': 'hash'}, {'openvswitch': {'other_config': {'tx-steering': 'thread'}}}, {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}):
        msg = _('Invalid value to --hints, see --help for valid values.')
        raise exceptions.CommandError(msg)