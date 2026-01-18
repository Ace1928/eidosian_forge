from __future__ import absolute_import
from .Nodes import *
from .ExprNodes import *
from .Errors import CompileError

    Tries to interpret a list of compile time option nodes.
    The result will be a tuple (optlist, optdict) but where
    all expression nodes have been interpreted. The result is
    in the form of tuples (value, pos).

    optlist is a list of nodes, while optdict is a DictNode (the
    result optdict is a dict)

    If type_env is set, all type nodes will be analysed and the resulting
    type set. Otherwise only interpretateable ExprNodes
    are allowed, other nodes raises errors.

    A CompileError will be raised if there are problems.
    