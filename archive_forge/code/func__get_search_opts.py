import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def _get_search_opts(self):
    search_opts = {'all_tenants': False, 'is_public': False, 'metadata': {}, 'extra_specs': {}, 'limit': None, 'name': None, 'status': None, 'host': None, 'share_server_id': None, 'share_network_id': None, 'share_type_id': None, 'snapshot_id': None, 'share_group_id': None, 'project_id': None, 'user_id': None, 'offset': None, 'is_soft_deleted': False, 'export_location': None, 'name~': None, 'description~': None}
    return search_opts