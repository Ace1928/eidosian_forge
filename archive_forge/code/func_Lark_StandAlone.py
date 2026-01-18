import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
def Lark_StandAlone(transformer=None, postlex=None):
    namespace = {'Rule': Rule, 'TerminalDef': TerminalDef}
    return Lark.deserialize(DATA, namespace, MEMO, transformer=transformer, postlex=postlex)