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
class NICAction(argparse.Action):

    def __init__(self, option_strings, dest, help=None, metavar=None, key=None):
        self.key = key
        super().__init__(option_strings=option_strings, dest=dest, nargs=None, const=None, default=[], type=None, choices=None, required=False, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, [])
        if self.key:
            if ',' in values or '=' in values:
                msg = _("Invalid argument %s; characters ',' and '=' are not allowed")
                raise argparse.ArgumentError(self, msg % values)
            values = '='.join([self.key, values])
        elif values in ('auto', 'none'):
            getattr(namespace, self.dest).append(values)
            return
        info = {'net-id': '', 'port-id': '', 'v4-fixed-ip': '', 'v6-fixed-ip': ''}
        for kv_str in values.split(','):
            k, sep, v = kv_str.partition('=')
            if k not in list(info) + ['tag'] or not v:
                msg = _("Invalid argument %s; argument must be of form 'net-id=net-uuid,port-id=port-uuid,v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr,tag=tag'")
                raise argparse.ArgumentError(self, msg % values)
            info[k] = v
        if info['net-id'] and info['port-id']:
            msg = _('Invalid argument %s; either network or port should be specified but not both')
            raise argparse.ArgumenteError(self, msg % values)
        getattr(namespace, self.dest).append(info)