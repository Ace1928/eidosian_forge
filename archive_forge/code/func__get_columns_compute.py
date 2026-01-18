from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _get_columns_compute(item):
    column_map = {}
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map)