import random
from abc import ABC, abstractmethod
class NoBackoff(ConstantBackoff):
    """No backoff upon failure"""

    def __init__(self):
        super().__init__(0)