from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, PY3
from ansible.module_utils.six.moves.http_client import responses as http_responses
class ModuleFailException(Exception):
    """
    If raised, module.fail_json() will be called with the given parameters after cleanup.
    """

    def __init__(self, msg, **args):
        super(ModuleFailException, self).__init__(self, msg)
        self.msg = msg
        self.module_fail_args = args

    def do_fail(self, module, **arguments):
        module.fail_json(msg=self.msg, other=self.module_fail_args, **arguments)