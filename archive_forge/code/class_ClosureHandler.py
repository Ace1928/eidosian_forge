import os
import threading
from queue import Empty as EmptyQueue, Queue
from torch._lazy.device_context import get_device_context
class ClosureHandler:

    def __init__(self):
        pass

    def run(self, closure):
        """Run closure function

        Args:
        closure: callable function to run
        """
        closure()

    def __call__(self, closures):
        for closure in closures:
            self.run(closure)