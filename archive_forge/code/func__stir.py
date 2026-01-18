import hashlib
import os
import random
import threading
import time
from typing import Any, Optional
def _stir(self, entropy: bytes) -> None:
    for c in entropy:
        if self.pool_index == self.hash_len:
            self.pool_index = 0
        b = c & 255
        self.pool[self.pool_index] ^= b
        self.pool_index += 1