from io import StringIO
from antlr4 import DFA
from antlr4.Utils import str_list
from antlr4.dfa.DFAState import DFAState
class LexerDFASerializer(DFASerializer):

    def __init__(self, dfa: DFA):
        super().__init__(dfa, None)

    def getEdgeLabel(self, i: int):
        return "'" + chr(i) + "'"