from __future__ import annotations
import math
def _getmedian(self):
    """Get median pixel level for each layer"""
    v = []
    for i in self.bands:
        s = 0
        half = self.count[i] // 2
        b = i * 256
        for j in range(256):
            s = s + self.h[b + j]
            if s > half:
                break
        v.append(j)
    return v