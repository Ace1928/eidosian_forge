from collections import deque
import select
import msgpack
def _process_input_notification(self):
    n = self._endpoint.get_notification()
    if n:
        self._notification_callback(n)