import argparse
import os.path
import sys
from os_ken import cfg
from os_ken import utils
from os_ken import version
def _get_argparse_kwargs(self, group, **kwargs):
    kwargs = cfg.MultiStrOpt._get_argparse_kwargs(self, group, **kwargs)
    kwargs['nargs'] = argparse.REMAINDER
    return kwargs