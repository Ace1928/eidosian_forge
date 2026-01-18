import re
import traceback
from oslo_log import log
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import client
from manilaclient.tests.functional import utils
@classmethod
def _determine_share_network_to_use(cls, client, share_type, microversion=None):
    """Determine what share network we need from the share type."""
    share_type = client.get_share_type(share_type, microversion=microversion)
    dhss_pattern = re.compile('driver_handles_share_servers : ([a-zA-Z]+)')
    dhss = dhss_pattern.search(share_type['required_extra_specs']).group(1)
    return client.share_network if dhss.lower() == 'true' else None