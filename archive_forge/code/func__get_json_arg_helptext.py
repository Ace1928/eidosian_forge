import sys
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import clusters as c_v1
def _get_json_arg_helptext(self):
    return '\n               JSON representation of the cluster scale object. Other\n               arguments (except for --wait) will not be taken into\n               account if this one is provided. Specifiying a JSON\n               object is also the only way to indicate specific\n               instances to decomission.\n               '