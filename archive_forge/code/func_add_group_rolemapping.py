from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def add_group_rolemapping(self, gid, cid, role_rep, realm='master'):
    """ Fetch the composite role of a client in a specified group on the Keycloak server.

        :param gid: ID of the group from which to obtain the rolemappings.
        :param cid: ID of the client from which to obtain the rolemappings.
        :param role_rep: Representation of the role to assign.
        :param realm: Realm from which to obtain the rolemappings.
        :return: None.
        """
    available_rolemappings_url = URL_CLIENT_GROUP_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, id=gid, client=cid)
    try:
        open_url(available_rolemappings_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(role_rep), validate_certs=self.validate_certs, timeout=self.connection_timeout)
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch available rolemappings for client %s in group %s, realm %s: %s' % (cid, gid, realm, str(e)))