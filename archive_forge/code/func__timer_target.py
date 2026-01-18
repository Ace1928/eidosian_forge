import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from multiprocessing import cpu_count
from typing import (
import pandas as pd
from tqdm.auto import tqdm
from cmdstanpy import (
from cmdstanpy.cmdstan_args import (
from cmdstanpy.stanfit import (
from cmdstanpy.utils import (
from cmdstanpy.utils.filesystem import temp_inits, temp_single_json
from . import progress as progbar
def _timer_target() -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    runset._set_timeout_flag(idx, True)