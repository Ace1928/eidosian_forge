import asyncore
from collections import deque
from warnings import _deprecated
def _collect_incoming_data(self, data):
    self.incoming.append(data)