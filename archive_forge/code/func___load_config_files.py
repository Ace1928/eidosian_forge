from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
@staticmethod
def __load_config_files(options: SharedCMDOptions, scratch_dir: str, ftype: str) -> T.List[str]:
    if ftype == 'cross':
        filenames = options.cross_file
    else:
        filenames = options.native_file
    if not filenames:
        return []
    found_invalid: T.List[str] = []
    missing: T.List[str] = []
    real: T.List[str] = []
    for i, f in enumerate(filenames):
        f = os.path.expanduser(os.path.expandvars(f))
        if os.path.exists(f):
            if os.path.isfile(f):
                real.append(os.path.abspath(f))
                continue
            elif os.path.isdir(f):
                found_invalid.append(os.path.abspath(f))
            else:
                copy = os.path.join(scratch_dir, f'{uuid.uuid4()}.{ftype}.ini')
                with open(f, encoding='utf-8') as rf:
                    with open(copy, 'w', encoding='utf-8') as wf:
                        wf.write(rf.read())
                real.append(copy)
                filenames[i] = copy
                continue
        if sys.platform != 'win32':
            paths = [os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))] + os.environ.get('XDG_DATA_DIRS', '/usr/local/share:/usr/share').split(':')
            for path in paths:
                path_to_try = os.path.join(path, 'meson', ftype, f)
                if os.path.isfile(path_to_try):
                    real.append(path_to_try)
                    break
            else:
                missing.append(f)
        else:
            missing.append(f)
    if missing:
        if found_invalid:
            mlog.log('Found invalid candidates for', ftype, 'file:', *found_invalid)
        mlog.log('Could not find any valid candidate for', ftype, 'files:', *missing)
        raise MesonException(f'Cannot find specified {ftype} file: {f}')
    return real