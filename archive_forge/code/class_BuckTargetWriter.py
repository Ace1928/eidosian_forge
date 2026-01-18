import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
class BuckTargetWriter:

    def __init__(self, filename):
        self.subdir, self.py_file = os.path.split(os.path.abspath(filename))
        self.target = self.py_file.replace('.py', '')
        self.path = f'{self.subdir.replace('/', '.')}.{self.target}'
        self.path = self.path[self.path.find('fbcode.'):]
        self.path = self.path[7:]
        tmp = self.subdir
        tmp = tmp[tmp.find('fbcode/'):][7:]
        self.cmd_line_path = f'//{tmp}:{self.target}'

    def build(self):
        extra_cpp_deps = '\n'.join([f'        "{x}",' for x in extra_deps])
        return textwrap.dedent(f'\nload("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")\n\npython_binary(\n    name="{self.target}",\n    srcs = ["{self.py_file}"],\n    compile = False,\n    deps = [\n        "//caffe2:torch",\n        "//caffe2/functorch:functorch",\n        "//triton:triton",\n        "{cur_target}",\n    ],\n    cpp_deps = [\n{extra_cpp_deps}\n    ],\n    main_module = "{self.path}",\n)\n')

    def write(self, print_msg=True):
        target_file = os.path.join(self.subdir, 'TARGETS')
        with open(target_file, 'w') as fd:
            fd.write(self.build())
        cmd_split = BUCK_CMD_PREFIX + [self.cmd_line_path]
        if print_msg:
            log.warning('Found an example that reproduces the error. Run this cmd to repro - %s', ' '.join(cmd_split))
        return cmd_split