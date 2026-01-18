from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _internal_parse(self, csource):
    ast, macros, csource = self._parse(csource)
    self._process_macros(macros)
    iterator = iter(ast.ext)
    for decl in iterator:
        if decl.name == '__dotdotdot__':
            break
    else:
        assert 0
    current_decl = None
    try:
        self._inside_extern_python = '__cffi_extern_python_stop'
        for decl in iterator:
            current_decl = decl
            if isinstance(decl, pycparser.c_ast.Decl):
                self._parse_decl(decl)
            elif isinstance(decl, pycparser.c_ast.Typedef):
                if not decl.name:
                    raise CDefError('typedef does not declare any name', decl)
                quals = 0
                if isinstance(decl.type.type, pycparser.c_ast.IdentifierType) and decl.type.type.names[-1].startswith('__dotdotdot'):
                    realtype = self._get_unknown_type(decl)
                elif isinstance(decl.type, pycparser.c_ast.PtrDecl) and isinstance(decl.type.type, pycparser.c_ast.TypeDecl) and isinstance(decl.type.type.type, pycparser.c_ast.IdentifierType) and decl.type.type.type.names[-1].startswith('__dotdotdot'):
                    realtype = self._get_unknown_ptr_type(decl)
                else:
                    realtype, quals = self._get_type_and_quals(decl.type, name=decl.name, partial_length_ok=True, typedef_example='*(%s *)0' % (decl.name,))
                self._declare('typedef ' + decl.name, realtype, quals=quals)
            elif decl.__class__.__name__ == 'Pragma':
                pass
            else:
                raise CDefError('unexpected <%s>: this construct is valid C but not valid in cdef()' % decl.__class__.__name__, decl)
    except CDefError as e:
        if len(e.args) == 1:
            e.args = e.args + (current_decl,)
        raise
    except FFIError as e:
        msg = self._convert_pycparser_error(e, csource)
        if msg:
            e.args = (e.args[0] + '\n    *** Err: %s' % msg,)
        raise