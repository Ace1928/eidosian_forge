import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def __check_children(self, name):
    return [_newSymbolTable(st, self._filename) for st in self._table.children if st.name == name]