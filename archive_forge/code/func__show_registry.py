import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
def _show_registry(registry):
    utils.print_dict(registry._info['registry'])