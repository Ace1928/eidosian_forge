from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
@python_2_unicode_compatible
class StatusValue(namedtuple('Status', 'value, is_pending')):
    MISSING = 'missing'
    OK = 'ok'
    NOT_MONITORED = 'not_monitored'
    INITIALIZING = 'initializing'
    DOES_NOT_EXIST = 'does_not_exist'
    EXECUTION_FAILED = 'execution_failed'
    ALL_STATUS = [MISSING, OK, NOT_MONITORED, INITIALIZING, DOES_NOT_EXIST, EXECUTION_FAILED]

    def __new__(cls, value, is_pending=False):
        return super(StatusValue, cls).__new__(cls, value, is_pending)

    def pending(self):
        return StatusValue(self.value, True)

    def __getattr__(self, item):
        if item in ('is_%s' % status for status in self.ALL_STATUS):
            return self.value == getattr(self, item[3:].upper())
        raise AttributeError(item)

    def __str__(self):
        return '%s%s' % (self.value, ' (pending)' if self.is_pending else '')