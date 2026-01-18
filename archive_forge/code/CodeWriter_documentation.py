from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode

        Temporaries are output like $1_1', where the first number is
        an index of the TempsBlockNode and the second number is an index
        of the temporary which that block allocates.
        