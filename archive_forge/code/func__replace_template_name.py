from __future__ import absolute_import, division, print_function
import re
import uuid
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _replace_template_name(self, template):
    """Replaces template name at runtime

        To allow us to do the switch-a-roo with temporary templates and
        checksum comparisons, we need to take the template provided to us
        and change its name to a temporary value so that BIG-IP will create
        a clone for us.

        :return string
        """
    pattern = 'sys\\s+application\\s+template\\s+[^ ]+'
    if self._values['name']:
        name = self._values['name']
    else:
        name = self._get_template_name()
    replace = 'sys application template {0}'.format(fq_name(self.partition, name))
    return re.sub(pattern, replace, template)