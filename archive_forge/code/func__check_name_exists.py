import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def _check_name_exists(self, collection, node):
    existing = collection.get(node.funcname)
    collection[node.funcname] = node
    if existing is not None and existing is not node and (node.is_block or existing.is_block):
        raise exceptions.CompileException("%%def or %%block named '%s' already exists in this template." % node.funcname, **node.exception_kwargs)