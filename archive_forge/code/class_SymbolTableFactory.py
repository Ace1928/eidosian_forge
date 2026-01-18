import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
class SymbolTableFactory:

    def __init__(self):
        self.__memo = weakref.WeakValueDictionary()

    def new(self, table, filename):
        if table.type == _symtable.TYPE_FUNCTION:
            return Function(table, filename)
        if table.type == _symtable.TYPE_CLASS:
            return Class(table, filename)
        return SymbolTable(table, filename)

    def __call__(self, table, filename):
        key = (table, filename)
        obj = self.__memo.get(key, None)
        if obj is None:
            obj = self.__memo[key] = self.new(table, filename)
        return obj