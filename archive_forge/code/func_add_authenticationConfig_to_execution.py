from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def add_authenticationConfig_to_execution(self, executionId, authenticationConfig, realm='master'):
    """ Add autenticatorConfig to the execution

        :param executionId: id of execution
        :param authenticationConfig: config to add to the execution
        :return: HTTPResponse object on success
        """
    try:
        open_url(URL_AUTHENTICATION_EXECUTION_CONFIG.format(url=self.baseurl, realm=realm, id=executionId), method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(authenticationConfig), timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to add authenticationConfig %s: %s' % (executionId, str(e)))