import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def _load_opts(self, opt):
    optfile = opt.get('init_opt')
    new_opt = Opt.load(optfile)
    for key, value in new_opt.items():
        if key not in opt:
            raise RuntimeError('Trying to set opt from file that does not exist: ' + str(key))
        if key not in opt['override']:
            opt[key] = value
            opt['override'][key] = value