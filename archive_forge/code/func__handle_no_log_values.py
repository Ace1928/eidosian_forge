from __future__ import absolute_import, division, print_function
import abc
import copy
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleFallbackNotFound, SEQUENCETYPE, remove_values
from ansible.module_utils.common._collections_compat import (
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.validation import (
from ansible.module_utils.common.text.formatters import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.action import ActionBase
def _handle_no_log_values(self, spec=None, param=None):
    if spec is None:
        spec = self.argument_spec
    if param is None:
        param = self.params
    try:
        self.no_log_values.update(list_no_log_values(spec, param))
    except TypeError as te:
        self.fail_json(msg='Failure when processing no_log parameters. Module invocation will be hidden. %s' % to_native(te), invocation={'module_args': 'HIDDEN DUE TO FAILURE'})
    for message in list_deprecations(spec, param):
        self.deprecate(message['msg'], version=message.get('version'), date=message.get('date'), collection_name=message.get('collection_name'))