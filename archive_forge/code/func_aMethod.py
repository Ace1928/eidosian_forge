import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
def aMethod(self):
    for i in range(10):
        self.x, self.y = (self.y, self.x)
        self.z = self.x + self.y
        assert self.z == 0, 'z == %d, not 0 as expected' % (self.z,)