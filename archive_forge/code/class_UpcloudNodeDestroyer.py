import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class UpcloudNodeDestroyer:
    """
    Helper class for destroying node.
    Node must be first stopped and then it can be
    destroyed

    :param  upcloud_node_operations: UpcloudNodeOperations instance
    :type   upcloud_node_operations: :class:`.UpcloudNodeOperations`

    :param  sleep_func: Callable function, which sleeps.
        Takes int argument to sleep in seconds (optional)
    :type   sleep_func: ``function``

    """
    WAIT_AMOUNT = 2
    SLEEP_COUNT_TO_TIMEOUT = 20

    def __init__(self, upcloud_node_operations, sleep_func=None):
        self._operations = upcloud_node_operations
        self._sleep_func = sleep_func or time.sleep
        self._sleep_count = 0

    def destroy_node(self, node_id):
        """
        Destroys the given node.

        :param  node_id: Id of the Node.
        :type   node_id: ``int``
        """
        self._stop_called = False
        self._sleep_count = 0
        return self._do_destroy_node(node_id)

    def _do_destroy_node(self, node_id):
        state = self._operations.get_node_state(node_id)
        if state == 'stopped':
            self._operations.destroy_node(node_id)
            return True
        elif state == 'error':
            return False
        elif state == 'started':
            if not self._stop_called:
                self._operations.stop_node(node_id)
                self._stop_called = True
            else:
                self._sleep()
            return self._do_destroy_node(node_id)
        elif state == 'maintenance':
            self._sleep()
            return self._do_destroy_node(node_id)
        elif state is None:
            return True

    def _sleep(self):
        if self._sleep_count > self.SLEEP_COUNT_TO_TIMEOUT:
            raise UpcloudTimeoutException('Timeout, could not destroy node')
        self._sleep_count += 1
        self._sleep_func(self.WAIT_AMOUNT)