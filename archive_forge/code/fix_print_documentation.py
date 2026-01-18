from lib2to3 import patcomp, pytree, fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Name, Call, Comma, String
Fixer for print.

Change:
    "print"          into "print()"
    "print ..."      into "print(...)"
    "print(...)"     not changed
    "print ... ,"    into "print(..., end=' ')"
    "print >>x, ..." into "print(..., file=x)"

No changes are applied if print_function is imported from __future__

