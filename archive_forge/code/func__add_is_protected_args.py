import argparse
from base64 import b64encode
import logging
import os
import sys
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack.image import image_signer
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.common import progressbar
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _add_is_protected_args(parser):
    protected_group = parser.add_mutually_exclusive_group()
    protected_group.add_argument('--protected', action='store_true', dest='is_protected', default=None, help=_('Prevent image from being deleted'))
    protected_group.add_argument('--unprotected', action='store_false', dest='is_protected', default=None, help=_('Allow image to be deleted (default)'))