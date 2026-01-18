from ast import (
import ast
import copy
from typing import Dict, Optional, Union
class Mangler(NodeTransformer):
    """
    Mangle given names in and ast tree to make sure they do not conflict with
    user code.
    """
    enabled: bool = True
    debug: bool = False

    def log(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __init__(self, predicate=None):
        if predicate is None:
            predicate = lambda name: name.startswith('___')
        self.predicate = predicate

    def visit_Name(self, node):
        if self.predicate(node.id):
            self.log('Mangling', node.id)
            node.id = 'mangle-' + node.id
        else:
            self.log('Not mangling', node.id)
        return node

    def visit_FunctionDef(self, node):
        if self.predicate(node.name):
            self.log('Mangling', node.name)
            node.name = 'mangle-' + node.name
        else:
            self.log('Not mangling', node.name)
        for arg in node.args.args:
            if self.predicate(arg.arg):
                self.log('Mangling function arg', arg.arg)
                arg.arg = 'mangle-' + arg.arg
            else:
                self.log('Not mangling function arg', arg.arg)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ImportFrom):
        return self._visit_Import_and_ImportFrom(node)

    def visit_Import(self, node: Import):
        return self._visit_Import_and_ImportFrom(node)

    def _visit_Import_and_ImportFrom(self, node: Union[Import, ImportFrom]):
        for alias in node.names:
            asname = alias.name if alias.asname is None else alias.asname
            if self.predicate(asname):
                new_name: str = 'mangle-' + asname
                self.log('Mangling Alias', new_name)
                alias.asname = new_name
            else:
                self.log('Not mangling Alias', alias.asname)
        return node