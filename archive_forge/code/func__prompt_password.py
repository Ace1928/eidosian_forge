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
def _prompt_password(self, verify=True):
    pw = None
    if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
        try:
            while True:
                pw1 = getpass.getpass('OS Password: ')
                if verify:
                    pw2 = getpass.getpass('Please verify: ')
                else:
                    pw2 = pw1
                if pw1 == pw2 and self._validate_string(pw1):
                    pw = pw1
                    break
        except EOFError:
            pass
    return pw