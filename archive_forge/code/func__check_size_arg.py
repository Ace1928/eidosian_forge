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
def _check_size_arg(args):
    """Check whether --size option is required or not.

    Require size parameter only in case when snapshot or source
    volume is not specified.
    """
    if (args.snapshot or args.source or args.backup) is None and args.size is None:
        msg = _('--size is a required option if snapshot, backup or source volume are not specified.')
        raise exceptions.CommandError(msg)