from __future__ import annotations
import math
def _getrms(self):
    """Get RMS for each layer"""
    return [math.sqrt(self.sum2[i] / self.count[i]) for i in self.bands]