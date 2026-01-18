from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def _generate_buffer_lookup_code(self, code, axes, cast_result=True):
    """
        Generate a single expression that indexes the memory view slice
        in each dimension.
        """
    bufp = self.buf_ptr
    type_decl = self.type.dtype.empty_declaration_code()
    for dim, index, access, packing in axes:
        shape = '%s.shape[%d]' % (self.cname, dim)
        stride = '%s.strides[%d]' % (self.cname, dim)
        suboffset = '%s.suboffsets[%d]' % (self.cname, dim)
        flag = get_memoryview_flag(access, packing)
        if flag in ('generic', 'generic_contiguous'):
            code.globalstate.use_utility_code(memviewslice_index_helpers)
            bufp = '__pyx_memviewslice_index_full(%s, %s, %s, %s)' % (bufp, index, stride, suboffset)
        elif flag == 'indirect':
            bufp = '(%s + %s * %s)' % (bufp, index, stride)
            bufp = '(*((char **) %s) + %s)' % (bufp, suboffset)
        elif flag == 'indirect_contiguous':
            bufp = '(*((char **) %s + %s) + %s)' % (bufp, index, suboffset)
        elif flag == 'strided':
            bufp = '(%s + %s * %s)' % (bufp, index, stride)
        else:
            assert flag == 'contiguous', flag
            bufp = '((char *) (((%s *) %s) + %s))' % (type_decl, bufp, index)
        bufp = '( /* dim=%d */ %s )' % (dim, bufp)
    if cast_result:
        return '((%s *) %s)' % (type_decl, bufp)
    return bufp