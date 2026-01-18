import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
class HipifyResult:

    def __init__(self, current_state, hipified_path):
        self.current_state = current_state
        self.hipified_path = hipified_path
        self.status = ''

    def __str__(self):
        return 'HipifyResult:: current_state: {}, hipified_path : {}, status: {}'.format(self.current_state, self.hipified_path, self.status)