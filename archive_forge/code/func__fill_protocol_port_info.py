import argparse
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import exceptions as nc_exc
def _fill_protocol_port_info(attrs, port_type, port_val):
    min_port, sep, max_port = port_val.partition(':')
    if not min_port:
        msg = "Invalid port value '%s', expected format is min-port:max-port or min-port."
        raise argparse.ArgumentTypeError(msg % port_val)
    if not max_port:
        max_port = min_port
    try:
        attrs[port_type + '_port_range_min'] = int(min_port)
        attrs[port_type + '_port_range_max'] = int(max_port)
    except ValueError:
        message = _('Protocol port value %s must be an integer or integer:integer.') % port_val
        raise nc_exc.CommandError(message=message)