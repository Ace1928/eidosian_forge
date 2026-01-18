from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _fused_signature_index(self, pyx_code):
    """
        Generate Cython code for constructing a persistent nested dictionary index of
        fused type specialization signatures.
        """
    pyx_code.put_chunk(u"\n                if not _fused_sigindex:\n                    for sig in <dict> signatures:\n                        sigindex_node = <dict> _fused_sigindex\n                        *sig_series, last_type = sig.strip('()').split('|')\n                        for sig_type in sig_series:\n                            if sig_type not in sigindex_node:\n                                sigindex_node[sig_type] = sigindex_node = {}\n                            else:\n                                sigindex_node = <dict> sigindex_node[sig_type]\n                        sigindex_node[last_type] = sig\n            ")