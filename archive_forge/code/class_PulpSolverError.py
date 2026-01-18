import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
class PulpSolverError(const.PulpError):
    """
    Pulp Solver-related exceptions
    """
    pass