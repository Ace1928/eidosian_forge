from __future__ import division
import datetime
import math
def _eta(self, maxval, currval, elapsed):
    return elapsed * maxval / float(currval) - elapsed