from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
def _login_vapi(self):
    """
        Login to vCenter API using REST call
        Returns: connection object

        """
    session = requests.Session()
    session.verify = self.validate_certs
    if not self.validate_certs:
        requests.packages.urllib3.disable_warnings()
    server = self.hostname
    if self.port:
        server += ':' + str(self.port)
    client, err = (None, None)
    try:
        client = create_vsphere_client(server=server, username=self.username, password=self.password, session=session)
    except Exception as error:
        err = error
    if client is None:
        msg = 'Failed to login to %s using %s' % (server, self.username)
        if err:
            msg += ' due to : %s' % to_native(err)
        raise AnsibleError(msg)
    return client