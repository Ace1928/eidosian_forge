import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def _combine_share_network_subnet_data(self, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None):
    """Combines params for share network subnet 'create' operation.

        :returns: text -- set of CLI parameters
        """
    data = dict()
    if neutron_net_id is not None:
        data['--neutron_net_id'] = neutron_net_id
    if neutron_subnet_id is not None:
        data['--neutron_subnet_id'] = neutron_subnet_id
    if availability_zone is not None:
        data['--availability_zone'] = availability_zone
    cmd = ''
    for key, value in data.items():
        cmd += '%(k)s=%(v)s ' % dict(k=key, v=value)
    return cmd