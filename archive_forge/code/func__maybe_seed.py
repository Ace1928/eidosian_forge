import os
import random
import time
from ._compat import long, binary_type
def _maybe_seed(self):
    if not self.seeded or self.seed_pid != os.getpid():
        try:
            seed = os.urandom(16)
        except Exception:
            try:
                r = open('/dev/urandom', 'rb', 0)
                try:
                    seed = r.read(16)
                finally:
                    r.close()
            except Exception:
                seed = str(time.time())
        self.seeded = True
        self.seed_pid = os.getpid()
        self.digest = None
        seed = bytearray(seed)
        self.stir(seed, True)