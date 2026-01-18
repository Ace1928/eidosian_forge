import json
import os
import re
import sys
import numpy as np
class DataSizeMapper:
    """For buffers, report the number of bytes."""

    def __call__(self, x):
        if x is not None:
            return '%d bytes' % len(x)
        else:
            return '--'