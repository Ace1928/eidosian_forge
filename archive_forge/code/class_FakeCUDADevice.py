import numpy as np
from collections import namedtuple
class FakeCUDADevice:

    def __init__(self):
        self.uuid = 'GPU-00000000-0000-0000-0000-000000000000'