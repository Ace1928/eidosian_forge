import argparse
import getpass
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from zunclient import api_versions
from zunclient import client as base_client
from zunclient.common.apiclient import auth
from zunclient.common import cliutils
from zunclient import exceptions as exc
from zunclient.i18n import _
from zunclient.v1 import shell as shell_v1
from zunclient import version
def _add_subparser_exclusive_args(self, subparser, exclusive_args, version, do_help, msg):
    for group_name, arguments in exclusive_args.items():
        if group_name == '__required__':
            continue
        required = exclusive_args['__required__'][group_name]
        exclusive_group = subparser.add_mutually_exclusive_group(required=required)
        self._add_subparser_args(exclusive_group, arguments, version, do_help, msg)