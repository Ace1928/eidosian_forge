import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def _format_kv(data):
    formatted_kv = {}
    values = data.split(',')
    for value in values:
        k, v = value.split('=')
        formatted_kv[k] = v
    return formatted_kv