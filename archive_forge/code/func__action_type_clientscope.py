from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def _action_type_clientscope(self, id=None, client_id=None, scope_type='default', realm='master', action='add'):
    """ Delete or add a clientscope of type.
        :param name: The name of the clientscope. A lookup will be performed to retrieve the clientscope ID.
        :param client_id: The ID of the clientscope (preferred to name).
        :param scope_type 'default' or 'optional'
        :param realm: The realm in which this group resides, default "master".
        """
    cid = None if client_id is None else self.get_client_id(client_id=client_id, realm=realm)
    clientscope_type_url = self._decide_url_type_clientscope(client_id, scope_type).format(realm=realm, id=id, cid=cid, url=self.baseurl)
    try:
        method = 'PUT' if action == 'add' else 'DELETE'
        return open_url(clientscope_type_url, method=method, http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        place = 'realm' if client_id is None else 'client ' + client_id
        self.fail_open_url(e, msg='Unable to %s %s clientscope %s @ %s : %s' % (action, scope_type, id, place, str(e)))