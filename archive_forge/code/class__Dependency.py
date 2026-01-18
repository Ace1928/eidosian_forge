from __future__ import absolute_import, division, print_function
import traceback
from contextlib import contextmanager
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
class _Dependency(object):
    _states = ['pending', 'failure', 'success']

    def __init__(self, name, reason=None, url=None, msg=None):
        self.name = name
        self.reason = reason
        self.url = url
        self.msg = msg
        self.state = 0
        self.trace = None
        self.exc = None

    def succeed(self):
        self.state = 2

    def fail(self, exc, trace):
        self.state = 1
        self.exc = exc
        self.trace = trace

    @property
    def message(self):
        if self.msg:
            return to_native(self.msg)
        else:
            return missing_required_lib(self.name, reason=self.reason, url=self.url)

    @property
    def failed(self):
        return self.state == 1

    def validate(self, module):
        if self.failed:
            module.fail_json(msg=self.message, exception=self.trace)

    def __str__(self):
        return '<dependency: {0} [{1}]>'.format(self.name, self._states[self.state])