from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def _handle_ssl_error(self, error):
    match = re.match("hostname.*doesn\\'t match (\\'.*\\')", str(error))
    if match:
        self.fail("You asked for verification that Docker daemons certificate's hostname matches %s. The actual certificate's hostname is %s. Most likely you need to set DOCKER_TLS_HOSTNAME or pass `tls_hostname` with a value of %s. You may also use TLS without verification by setting the `tls` parameter to true." % (self.auth_params['tls_hostname'], match.group(1), match.group(1)))
    self.fail('SSL Exception: %s' % error)