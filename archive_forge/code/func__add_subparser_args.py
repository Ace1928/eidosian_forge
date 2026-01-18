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
def _add_subparser_args(self, subparser, arguments, version, do_help, msg):
    for args, kwargs in arguments:
        start_version = kwargs.get('start_version', None)
        if start_version:
            start_version = api_versions.APIVersion(start_version)
            end_version = kwargs.get('end_version', None)
            if end_version:
                end_version = api_versions.APIVersion(end_version)
            else:
                end_version = api_versions.APIVersion('%s.latest' % start_version.ver_major)
            if do_help:
                kwargs['help'] = kwargs.get('help', '') + msg % {'start': start_version.get_string(), 'end': end_version.get_string()}
            elif not version.matches(start_version, end_version):
                continue
        kw = kwargs.copy()
        kw.pop('start_version', None)
        kw.pop('end_version', None)
        subparser.add_argument(*args, **kwargs)