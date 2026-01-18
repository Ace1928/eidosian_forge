from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def _precision_warn(p1, p2, extra=''):
    t = 'Lossy conversion from {} to {}. {} Convert image to {} prior to saving to suppress this warning.'
    logger.warning(t.format(p1, p2, extra, p2))