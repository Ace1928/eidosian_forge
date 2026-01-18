from pythran.passmanager import Transformation
import pythran.metadata as metadata
from pythran.spec import parse_pytypes
from pythran.types.conversion import pytype_to_ctype
from pythran.utils import isstr
from gast import AST
import gast as ast
import re
class OMPDirective(AST):
    """Turn a string into a context-dependent metadata.
    >>> o = OMPDirective("omp for private(a,b) shared(c)")
    >>> o.s
    'omp for private({},{}) shared({})'
    >>> [ type(dep) for dep in o.deps ]
    [<class 'gast.gast.Name'>, <class 'gast.gast.Name'>, <class 'gast.gast.Name'>]
    >>> [ dep.id for dep in o.deps ]
    ['a', 'b', 'c']
    """

    def __init__(self, *args):
        super(OMPDirective, self).__init__()
        if not args:
            return
        self.deps = []
        self.private_deps = []
        self.shared_deps = []

        def tokenize(s):
            """A simple contextual "parser" for an OpenMP string"""
            out = ''
            par_count = 0
            curr_index = 0
            in_reserved_context = False
            in_declare = False
            in_shared = in_private = False
            while curr_index < len(s):
                bounds = []
                if in_declare and is_declare_typename(s, curr_index, bounds):
                    start, stop = bounds
                    pytypes = parse_pytypes(s[start:stop])
                    out += ', '.join(map(pytype_to_ctype, pytypes))
                    curr_index = stop
                    continue
                m = re.match('^([a-zA-Z_]\\w*)', s[curr_index:])
                if m:
                    word = m.group(0)
                    curr_index += len(word)
                    if in_reserved_context or (in_declare and word in declare_keywords) or (par_count == 0 and word in keywords):
                        out += word
                        in_reserved_context = word in reserved_contex
                        in_declare |= word == 'declare'
                        in_private |= word == 'private'
                        in_shared |= word == 'shared'
                    else:
                        out += '{}'
                        self.deps.append(ast.Name(word, ast.Load(), None, None))
                        isattr = re.match('^\\s*(\\.\\s*[a-zA-Z_]\\w*)', s[curr_index:])
                        if isattr:
                            attr = isattr.group(0)
                            curr_index += len(attr)
                            self.deps[-1] = ast.Attribute(self.deps[-1], attr[1:], ast.Load())
                        if in_private:
                            self.private_deps.append(self.deps[-1])
                        if in_shared:
                            self.shared_deps.append(self.deps[-1])
                elif s[curr_index] == '(':
                    par_count += 1
                    curr_index += 1
                    out += '('
                elif s[curr_index] == ')':
                    par_count -= 1
                    curr_index += 1
                    out += ')'
                    if par_count == 0:
                        in_reserved_context = False
                        in_shared = in_private = False
                else:
                    if s[curr_index] in ',:':
                        in_reserved_context = False
                    out += s[curr_index]
                    curr_index += 1
            return out
        self.s = tokenize(args[0])
        self._fields = ('deps', 'shared_deps', 'private_deps')