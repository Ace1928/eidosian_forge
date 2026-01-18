from parso.python import tree
from parso.python.token import PythonTokenTypes
from parso.parser import BaseParser
def _recovery_tokenize(self, tokens):
    for token in tokens:
        typ = token[0]
        if typ == DEDENT:
            o = self._omit_dedent_list
            if o and o[-1] == self._indent_counter:
                o.pop()
                self._indent_counter -= 1
                continue
            self._indent_counter -= 1
        elif typ == INDENT:
            self._indent_counter += 1
        yield token