from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def add_default_clientscope(self, id, realm='master', client_id=None):
    """Add a client scope as default either on realm or client level.

        :param id: Client scope Id.
        :param realm: Realm in which the clientscope resides.
        :param client_id: The client in which the clientscope resides.
        """
    self._action_type_clientscope(id, client_id, 'default', realm, 'add')