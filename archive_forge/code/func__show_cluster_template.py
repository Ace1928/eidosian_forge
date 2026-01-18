from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
def _show_cluster_template(cluster_template):
    del cluster_template._info['links']
    for field in cluster_template._info:
        if cluster_template._info[field] is None:
            setattr(cluster_template, field, '-')
    columns = CLUSTER_TEMPLATE_ATTRIBUTES
    return (columns, osc_utils.get_item_properties(cluster_template, columns))