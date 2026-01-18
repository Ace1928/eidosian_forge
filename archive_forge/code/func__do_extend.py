import collections
import ipaddress
from oslo_utils import uuidutils
import re
import string
from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient.v2 import share_instances
def _do_extend(self, share, new_size, action_name, force=False):
    """Extend the size of the specified share.

        :param share: either share object or text with its ID.
        :param new_size: The desired size to extend share to.
        :param force: if set to True, the scheduler's capacity decisions are
                      not accounted for. Setting this parameter to True does
                      not mean that the request will always succeed.
        """
    req_body = {'new_size': new_size}
    if force:
        req_body['force'] = 'true'
    return self._action(action_name, share, req_body)