from __future__ import annotations
from typing import List, Optional, Union, Any, Set
import os
import glob
import json
from pathlib import Path
from pkg_resources import resource_filename
import pooch
from .exceptions import ParameterError
def __get_files(dir_name: Union[str, os.PathLike[Any]], extensions: Set[str]):
    """Get a list of files in a single directory"""
    dir_name = os.path.abspath(os.path.expanduser(dir_name))
    myfiles = set()
    for sub_ext in extensions:
        globstr = os.path.join(dir_name, '*' + os.path.extsep + sub_ext)
        myfiles |= set(glob.glob(globstr))
    return myfiles