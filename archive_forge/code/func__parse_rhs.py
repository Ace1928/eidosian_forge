from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
def _parse_rhs(self):
    a, z = self._parse_items()
    if self.value != '|':
        return (a, z)
    else:
        aa = NFAState(self._current_rule_name)
        zz = NFAState(self._current_rule_name)
        while True:
            aa.add_arc(a)
            z.add_arc(zz)
            if self.value != '|':
                break
            self._gettoken()
            a, z = self._parse_items()
        return (aa, zz)