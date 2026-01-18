from itertools import chain
from sympy.codegen.ast import Type, none
from .c import C89CodePrinter, C99CodePrinter
from sympy.printing.codeprinter import cxxcode # noqa:F401
class CXX11CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++11'
    reserved_words = set(reserved['C++11'])
    type_mappings = dict(chain(CXX98CodePrinter.type_mappings.items(), {Type('int8'): ('int8_t', {'cstdint'}), Type('int16'): ('int16_t', {'cstdint'}), Type('int32'): ('int32_t', {'cstdint'}), Type('int64'): ('int64_t', {'cstdint'}), Type('uint8'): ('uint8_t', {'cstdint'}), Type('uint16'): ('uint16_t', {'cstdint'}), Type('uint32'): ('uint32_t', {'cstdint'}), Type('uint64'): ('uint64_t', {'cstdint'}), Type('complex64'): ('std::complex<float>', {'complex'}), Type('complex128'): ('std::complex<double>', {'complex'}), Type('bool'): ('bool', None)}.items()))

    def _print_using(self, expr):
        if expr.alias == none:
            return super()._print_using(expr)
        else:
            return 'using %(alias)s = %(type)s' % expr.kwargs(apply=self._print)