from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
@staticmethod
def _check_attrs(attrs):
    verify_args = ['vip_subnet_id', 'vip_network_id', 'vip_port_id']
    if not any((i in attrs for i in verify_args)):
        msg = 'Missing required argument: Requires one of --vip-subnet-id, --vip-network-id or --vip-port-id'
        raise exceptions.CommandError(msg)
    if all((i in attrs for i in ('vip_network_id', 'vip_port_id'))):
        msg = 'Argument error: --vip-port-id can not be used with --vip-network-id'
        raise exceptions.CommandError(msg)