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
def _expand_port_hint_aliases(hints):
    if hints == {'ovs-tx-steering': 'thread'}:
        return {'openvswitch': {'other_config': {'tx-steering': 'thread'}}}
    elif hints == {'ovs-tx-steering': 'hash'}:
        return {'openvswitch': {'other_config': {'tx-steering': 'hash'}}}
    else:
        return hints