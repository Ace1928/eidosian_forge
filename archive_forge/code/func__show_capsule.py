import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
def _show_capsule(capsule):
    zun_utils.format_container_addresses(capsule)
    utils.print_dict(capsule._info)