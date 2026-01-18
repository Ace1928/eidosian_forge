import glob
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib import filesystem
from wandb.viz import CustomChart
from . import run as internal_run
def _calculate_namespace(self, logdir: str, rootdir: str) -> Optional[str]:
    namespace: Optional[str]
    dirs = list(self._logdirs) + [logdir]
    if os.path.isfile(logdir):
        filename = os.path.basename(logdir)
    else:
        filename = ''
    if rootdir == '':
        rootdir = util.to_forward_slash_path(os.path.dirname(os.path.commonprefix(dirs)))
        namespace = logdir.replace(filename, '').replace(rootdir, '').strip('/')
        if len(dirs) == 1 and namespace not in ['train', 'validation']:
            namespace = None
    else:
        namespace = logdir.replace(filename, '').replace(rootdir, '').strip('/')
    return namespace