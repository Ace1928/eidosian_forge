from weakref import ref as weakref_ref
def addSymbol(self, obj, symb):
    """
        Add a symbol for a given object

        This method assumes that objects and symbol names will not conflict.
        """
    nSymbols = len(self.byObject) + 1
    self.byObject[id(obj)] = symb
    self.bySymbol[symb] = obj
    if nSymbols != len(self.bySymbol):
        raise RuntimeError('SymbolMap.addSymbol(): duplicate symbol.  SymbolMap likely in an inconsistent state')
    if len(self.byObject) != len(self.bySymbol):
        raise RuntimeError('SymbolMap.addSymbol(): duplicate object.  SymbolMap likely in an inconsistent state')