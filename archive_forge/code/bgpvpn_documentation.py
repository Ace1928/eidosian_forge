import logging
from osc_lib.cli import format_columns
from osc_lib.cli.parseractions import KeyValueAction
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
Adds to parser arguments common to create, set and unset commands.

    :params ArgumentParser parser: argparse object contains all command's
                                   arguments
    :params string update: Determines if it is a create command (value: None),
                           it is a set command (value: 'set') or if it is an
                           unset command (value: 'unset')
    