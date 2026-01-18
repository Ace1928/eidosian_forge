import datetime
import functools
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _formatters(project_cache):
    return {'project_id': functools.partial(ProjectColumn, project_cache=project_cache), 'server_usages': CountColumn, 'total_memory_mb_usage': FloatColumn, 'total_vcpus_usage': FloatColumn, 'total_local_gb_usage': FloatColumn}