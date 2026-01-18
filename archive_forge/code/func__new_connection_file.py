import atexit
import os
import signal
import sys
import typing as t
import uuid
import warnings
from jupyter_core.application import base_aliases, base_flags
from traitlets import CBool, CUnicode, Dict, List, Type, Unicode
from traitlets.config.application import boolean_flag
from . import KernelManager, connect, find_connection_file, tunnel_to_kernel
from .blocking import BlockingKernelClient
from .connect import KernelConnectionInfo
from .kernelspec import NoSuchKernel
from .localinterfaces import localhost
from .restarter import KernelRestarter
from .session import Session
from .utils import _filefind
def _new_connection_file(self) -> str:
    cf = ''
    while not cf:
        ident = str(uuid.uuid4()).split('-')[-1]
        runtime_dir = self.runtime_dir
        cf = os.path.join(runtime_dir, 'kernel-%s.json' % ident)
        cf = cf if not os.path.exists(cf) else ''
    return cf