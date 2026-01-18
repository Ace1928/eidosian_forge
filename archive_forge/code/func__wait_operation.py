import time
import hashlib
from libcloud.utils.py3 import b
from libcloud.common.base import ConnectionKey
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
def _wait_operation(self, id, timeout=DEFAULT_TIMEOUT, check_interval=DEFAULT_INTERVAL):
    """Wait for an operation to succeed"""
    for i in range(0, timeout, check_interval):
        try:
            op = self.connection.request('operation.info', int(id)).object
            if op['step'] == 'DONE':
                return True
            if op['step'] in ['ERROR', 'CANCEL']:
                return False
        except (KeyError, IndexError):
            pass
        except Exception as e:
            raise GandiException(1002, e)
        time.sleep(check_interval)
    return False