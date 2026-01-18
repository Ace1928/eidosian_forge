from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.plugins import AnsiblePlugin
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_text
def _alarm_handler(self, signum, frame):
    """Alarm handler raised in case of command timeout """
    self._connection.queue_message('log', 'closing shell due to command timeout (%s seconds).' % self._connection._play_context.timeout)
    self.close()