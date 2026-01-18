import threading
from itertools import count
class RestoreState:

    def __init__(self, context, state_id):
        self.state_id = state_id
        self.context = context
        self.restored = False

    def __enter__(self):
        return self.context

    def __exit__(self, _exc_type, _exc_value, _exc_tb):
        self.restore()

    def restore(self):
        if self.restored:
            return
        self.context._restore(self.state_id)
        self.restored = True