from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
def _parse_atom(self):
    if self.value == '(':
        self._gettoken()
        a, z = self._parse_rhs()
        self._expect(PythonTokenTypes.OP, ')')
        return (a, z)
    elif self.type in (PythonTokenTypes.NAME, PythonTokenTypes.STRING):
        a = NFAState(self._current_rule_name)
        z = NFAState(self._current_rule_name)
        a.add_arc(z, self.value)
        self._gettoken()
        return (a, z)
    else:
        self._raise_error('expected (...) or NAME or STRING, got %s/%s', self.type, self.value)