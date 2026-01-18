import re
from typing import Optional
import numpy as np
import ply.lex as lex
from cirq.contrib.qasm_import.exception import QasmException
class QasmLexer:

    def __init__(self):
        self.lex = lex.lex(object=self, debug=False)
    literals = '{}[]();,+/*-^'
    reserved = {'qreg': 'QREG', 'creg': 'CREG', 'measure': 'MEASURE', 'if': 'IF', '->': 'ARROW', '==': 'EQ'}
    tokens = ['FORMAT_SPEC', 'NUMBER', 'NATURAL_NUMBER', 'QELIBINC', 'ID', 'PI'] + list(reserved.values())

    def t_newline(self, t):
        """\\n+"""
        t.lexer.lineno += len(t.value)
    t_ignore = ' \t'

    def t_PI(self, t):
        """pi"""
        t.value = np.pi
        return t

    def t_NUMBER(self, t):
        """(
        (
        [0-9]+\\.?|
        [0-9]?\\.[0-9]+
        )
        [eE][+-]?[0-9]+
        )|
        (
        ([0-9]+)?\\.[0-9]+|
        [0-9]+\\.)"""
        t.value = float(t.value)
        return t

    def t_NATURAL_NUMBER(self, t):
        """\\d+"""
        t.value = int(t.value)
        return t

    def t_FORMAT_SPEC(self, t):
        """OPENQASM(\\s+)([^\\s\\t\\;]*);"""
        match = re.match('OPENQASM(\\s+)([^\\s\\t;]*);', t.value)
        t.value = match.groups()[1]
        return t

    def t_QELIBINC(self, t):
        """include(\\s+)"qelib1.inc";"""
        return t

    def t_QREG(self, t):
        """qreg"""
        return t

    def t_CREG(self, t):
        """creg"""
        return t

    def t_MEASURE(self, t):
        """measure"""
        return t

    def t_IF(self, t):
        """if"""
        return t

    def t_ARROW(self, t):
        """->"""
        return t

    def t_EQ(self, t):
        """=="""
        return t

    def t_ID(self, t):
        """[a-zA-Z][a-zA-Z\\d_]*"""
        return t

    def t_COMMENT(self, t):
        """//.*"""

    def t_error(self, t):
        raise QasmException(f"Illegal character '{t.value[0]}' at line {t.lineno}")

    def input(self, qasm):
        self.lex.input(qasm)

    def token(self) -> Optional[lex.Token]:
        return self.lex.token()