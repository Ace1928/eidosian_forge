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
def _deferredAttr(self, attr):
    return DeferredObjectProxy(self, attr)