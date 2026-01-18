import os
import re
import sys
from typing import List
Checks all pytorch code for CUDA kernel launches without cuda error checks

    Returns:
        The number of unsafe kernel launches in the codebase
    