import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def CheckNode(node, keypath):
    if isinstance(node, ast.Dict):
        dict = {}
        for key, value in zip(node.keys, node.values):
            assert isinstance(key, ast.Str)
            key = key.s
            if key in dict:
                raise GypError("Key '" + key + "' repeated at level " + repr(len(keypath) + 1) + " with key path '" + '.'.join(keypath) + "'")
            kp = list(keypath)
            kp.append(key)
            dict[key] = CheckNode(value, kp)
        return dict
    elif isinstance(node, ast.List):
        children = []
        for index, child in enumerate(node.elts):
            kp = list(keypath)
            kp.append(repr(index))
            children.append(CheckNode(child, kp))
        return children
    elif isinstance(node, ast.Str):
        return node.s
    else:
        raise TypeError("Unknown AST node at key path '" + '.'.join(keypath) + "': " + repr(node))