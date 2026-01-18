import argparse
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
from cmdstanpy.utils.cmdstan import get_download_url
from . import progress as progbar
class InstallationSettings:
    """
    A static installation settings object
    """

    def __init__(self, *, version: Optional[str]=None, dir: Optional[str]=None, progress: bool=False, verbose: bool=False, overwrite: bool=False, cores: int=1, compiler: bool=False, **kwargs: Any):
        self.version = version if version else latest_version()
        self.dir = dir if dir else home_cmdstan()
        self.progress = progress
        self.verbose = verbose
        self.overwrite = overwrite
        self.cores = cores
        self.compiler = compiler and is_windows()
        _ = kwargs