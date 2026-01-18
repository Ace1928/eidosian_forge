import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def _getProxyOption(self, opt):
    val = self._proxyOptions[opt]
    if val is None:
        return self._handler.getProxyOption(opt)
    return val