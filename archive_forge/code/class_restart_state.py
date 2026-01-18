import os
import signal
import sys
import pickle
from .exceptions import RestartFreqExceeded
from time import monotonic
from io import BytesIO
class restart_state:
    RestartFreqExceeded = RestartFreqExceeded

    def __init__(self, maxR, maxT):
        self.maxR, self.maxT = (maxR, maxT)
        self.R, self.T = (0, None)

    def step(self, now=None):
        now = monotonic() if now is None else now
        R = self.R
        if self.T and now - self.T >= self.maxT:
            self.T, self.R = (now, 0)
        elif self.maxR and self.R >= self.maxR:
            if self.R:
                self.R = 0
                raise self.RestartFreqExceeded('%r in %rs' % (R, self.maxT))
        if self.T is None:
            self.T = now
        self.R += 1