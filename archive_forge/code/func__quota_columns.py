from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.i18n import _
def _quota_columns(quota):
    return quota._info.keys()