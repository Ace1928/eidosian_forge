import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
class OrderedSymbolTable(SymbolTable):

    def __init__(self):
        self.scopes_ = [{}]

    def enter_scope(self):
        self.scopes_.append({})

    def resolve(self, name, case_insensitive=False):
        SymbolTable.resolve(self, name, case_insensitive=case_insensitive)

    def range(self, start, end):
        for scope in reversed(self.scopes_):
            if start in scope and end in scope:
                start_idx = list(scope.keys()).index(start)
                end_idx = list(scope.keys()).index(end)
                return list(scope.keys())[start_idx:end_idx + 1]
        return None