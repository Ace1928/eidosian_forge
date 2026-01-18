from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc import utils
from saharaclient.osc.v1 import job_binaries as jb_v1
class ListJobBinaries(jb_v1.ListJobBinaries):
    """Lists job binaries"""
    log = logging.getLogger(__name__ + '.ListJobBinaries')