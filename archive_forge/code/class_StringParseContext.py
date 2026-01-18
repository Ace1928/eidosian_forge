from __future__ import absolute_import
import re
from io import StringIO
from .Scanning import PyrexScanner, StringSourceDescriptor
from .Symtab import ModuleScope
from . import PyrexTypes
from .Visitor import VisitorTransform
from .Nodes import Node, StatListNode
from .ExprNodes import NameNode
from .StringEncoding import _unicode
from . import Parsing
from . import Main
from . import UtilNodes
class StringParseContext(Main.Context):

    def __init__(self, name, include_directories=None, compiler_directives=None, cpp=False):
        if include_directories is None:
            include_directories = []
        if compiler_directives is None:
            compiler_directives = {}
        Main.Context.__init__(self, include_directories, compiler_directives, cpp=cpp, language_level='3str')
        self.module_name = name

    def find_module(self, module_name, from_module=None, pos=None, need_pxd=1, absolute_fallback=True, relative_import=False):
        if module_name not in (self.module_name, 'cython'):
            raise AssertionError('Not yet supporting any cimports/includes from string code snippets')
        return ModuleScope(module_name, parent_module=None, context=self)