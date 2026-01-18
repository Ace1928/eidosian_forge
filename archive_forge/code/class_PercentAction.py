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
class PercentAction(argparse.Action):

    def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None):
        if nargs == 0:
            raise ValueError('nargs for store actions must be != 0; if you have nothing to store, actions such as store true or store const may be more appropriate')
        if const is not None:
            raise ValueError('const does not make sense for PercentAction')
        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default, type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        x = int(values)
        if not 0 < x <= 100:
            raise argparse.ArgumentError(self, 'Must be between 0 and 100')
        setattr(namespace, self.dest, x)