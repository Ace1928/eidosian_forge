from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
class FilterModuleMock(object):

    def __init__(self, params):
        self.check_mode = True
        self.params = params
        self._diff = False

    def fail_json(self, msg, **kwargs):
        raise AnsibleFilterError(msg)