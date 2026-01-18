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
class _HelpAllAction(argparse._HelpAction):

    def __call__(self, parser, namespace, values, option_string=None):
        if hasattr(parser, '_unsuppress_hidden'):
            parser._unsuppress_hidden()
        super().__call__(parser, namespace, values, option_string=option_string)