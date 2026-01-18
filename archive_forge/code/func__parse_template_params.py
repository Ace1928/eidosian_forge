import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
def _parse_template_params(cli_param_list):
    return dict((cli_param.split('=', 1) for cli_param in cli_param_list)) if cli_param_list else {}