from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def _decide_url_type_clientscope(self, client_id=None, scope_type='default'):
    """Decides which url to use.
        :param scope_type this can be either optional or default
        :param client_id: The client in which the clientscope resides.
        """
    if client_id is None:
        if scope_type == 'default':
            return URL_DEFAULT_CLIENTSCOPE
        if scope_type == 'optional':
            return URL_OPTIONAL_CLIENTSCOPE
    else:
        if scope_type == 'default':
            return URL_CLIENT_DEFAULT_CLIENTSCOPE
        if scope_type == 'optional':
            return URL_CLIENT_OPTIONAL_CLIENTSCOPE