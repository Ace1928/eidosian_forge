import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
class NotThisMethod(Exception):
    """Exception raised if a method is not valid for the current scenario."""