import os
import hashlib
import pickle
import time
import shutil
import glob
from ..interfaces.base import BaseInterface
from ..pipeline.engine import Node
from ..pipeline.engine.utils import modify_paths
class _MemoryCallback(object):
    """An object to avoid closures and have everything pickle"""

    def __init__(self, memory):
        self.memory = memory

    def __call__(self, dir_name, job_name):
        self.memory._log_name(dir_name, job_name)