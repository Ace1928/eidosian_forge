import copy
import math
import numpy
def AddPoints(self, pts, names):
    if len(pts) != len(names):
        raise ValueError('input length mismatch')
    self.data += pts
    self.ptNames += names
    self.nPts = len(self.data)