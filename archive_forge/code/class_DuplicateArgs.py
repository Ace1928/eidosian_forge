import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
class DuplicateArgs(Exception):
    """More than one of the same argument type was passed."""

    def __init__(self, param, dupes):
        msg = _('Duplicate "%(param)s" arguments: %(dupes)s') % {'param': param, 'dupes': ', '.join(dupes)}
        super(DuplicateArgs, self).__init__(msg)