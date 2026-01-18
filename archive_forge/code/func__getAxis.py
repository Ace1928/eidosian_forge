import copy
import os
import pickle
import warnings
import numpy as np
def _getAxis(self, name):
    for i in range(0, len(self._info)):
        axis = self._info[i]
        if 'name' in axis and axis['name'] == name:
            return i
    raise Exception('No axis named %s.\n  info=%s' % (name, self._info))