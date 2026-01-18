import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _delete_share_and_share_server(self, share_id, share_server_id):
    self.client.delete_share(share_id)
    self.client.wait_for_share_deletion(share_id)
    self.client.delete_share_server(share_server_id)
    self.client.wait_for_share_server_deletion(share_server_id)