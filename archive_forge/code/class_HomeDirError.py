import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
class HomeDirError(Exception):
    pass