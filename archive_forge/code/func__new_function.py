import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def _new_function(self, code):
    name = 'alias_%d' % self._count
    self._count += 1
    self.alias_js_code[name] = code
    return name