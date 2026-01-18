from itertools import chain
from sympy.codegen.ast import Type, none
from .c import C89CodePrinter, C99CodePrinter
from sympy.printing.codeprinter import cxxcode # noqa:F401
class CXX98CodePrinter(_CXXCodePrinterBase, C89CodePrinter):
    standard = 'C++98'
    reserved_words = set(reserved['C++98'])