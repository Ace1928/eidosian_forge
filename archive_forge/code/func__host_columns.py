from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
def _host_columns(host):
    return host._info.keys()