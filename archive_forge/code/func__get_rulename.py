import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def _get_rulename(name):
    name = {'_': '_ws_maybe', '__': '_ws'}.get(name, name)
    return 'n_' + name.replace('$', '__DOLLAR__').lower()