import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_r3900(self):
    return self.__cputype(3900)