from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def add_group_realm_rolemapping(self, gid, role_rep, realm='master'):
    """ Add the specified realm role to specified group on the Keycloak server.

        :param gid: ID of the group to add the role mapping.
        :param role_rep: Representation of the role to assign.
        :param realm: Realm from which to obtain the rolemappings.
        :return: None.
        """
    url = URL_REALM_GROUP_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, group=gid)
    try:
        open_url(url, method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(role_rep), validate_certs=self.validate_certs, timeout=self.connection_timeout)
    except Exception as e:
        self.fail_open_url(e, msg='Could add realm role mappings for group %s, realm %s: %s' % (gid, realm, str(e)))