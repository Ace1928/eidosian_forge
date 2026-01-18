from sympy.printing.codeprinter import CodePrinter
from sympy.core import symbols
from sympy.core.symbol import Dummy
from sympy.testing.pytest import raises
class CrashingCodePrinter(CodePrinter):

    def emptyPrinter(self, obj):
        raise NotImplementedError