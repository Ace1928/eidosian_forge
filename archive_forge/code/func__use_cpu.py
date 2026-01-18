import os
import numpy as np
import threading
from time import time
from .. import config, logging
def _use_cpu(x):
    ctr = 0
    while ctr < 10000000.0:
        ctr += 1
        x * x