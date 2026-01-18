import os
import sys
import threading
from . import process
from . import reduction
class ProcessError(Exception):
    pass