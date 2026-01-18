from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _wait_for_completion(profitbricks, promise, wait_timeout, msg):
    if not promise:
        return
    wait_timeout = time.time() + wait_timeout
    while wait_timeout > time.time():
        time.sleep(5)
        operation_result = profitbricks.get_request(request_id=promise['requestId'], status=True)
        if operation_result['metadata']['status'] == 'DONE':
            return
        elif operation_result['metadata']['status'] == 'FAILED':
            raise Exception('Request failed to complete ' + msg + ' "' + str(promise['requestId']) + '" to complete.')
    raise Exception('Timed out waiting for async operation ' + msg + ' "' + str(promise['requestId']) + '" to complete.')