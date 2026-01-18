from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
import traceback
class RedisAnsible(object):
    """Base class for Redis module"""

    def __init__(self, module):
        self.module = module
        self.connection = self._connect()

    def _connect(self):
        try:
            return Redis(**redis_auth_params(self.module))
        except Exception as e:
            self.module.fail_json(msg='{0}'.format(str(e)))
        return None