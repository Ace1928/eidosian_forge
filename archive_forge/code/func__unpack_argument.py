from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def _unpack_argument(self, pyx_code):
    pyx_code.put_chunk(u'\n                # PROCESSING ARGUMENT {{arg_tuple_idx}}\n                if {{arg_tuple_idx}} < len(<tuple>args):\n                    arg = (<tuple>args)[{{arg_tuple_idx}}]\n                elif kwargs is not None and \'{{arg.name}}\' in <dict>kwargs:\n                    arg = (<dict>kwargs)[\'{{arg.name}}\']\n                else:\n                {{if arg.default}}\n                    arg = (<tuple>defaults)[{{default_idx}}]\n                {{else}}\n                    {{if arg_tuple_idx < min_positional_args}}\n                        raise TypeError("Expected at least %d argument%s, got %d" % (\n                            {{min_positional_args}}, {{\'"s"\' if min_positional_args != 1 else \'""\'}}, len(<tuple>args)))\n                    {{else}}\n                        raise TypeError("Missing keyword-only argument: \'%s\'" % "{{arg.default}}")\n                    {{endif}}\n                {{endif}}\n            ')