import collections
import re
import uuid
from ply import lex
from ply import yacc
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import lexer
from yaql.language import parser
from yaql.language import utils
def _build_operator_table(self, name_generator):
    operators = {}
    name_value_op = None
    precedence = 1
    for record in self.operators:
        if not record:
            precedence += 1
            continue
        up, bp, name, alias = operators.get(record[0], (0, 0, '', None))
        if record[1] == OperatorType.NAME_VALUE_PAIR:
            if name_value_op is not None:
                raise exceptions.InvalidOperatorTableException(record[0])
            name_value_op = record[0]
            continue
        if record[1] == OperatorType.PREFIX_UNARY:
            if up:
                raise exceptions.InvalidOperatorTableException(record[0])
            up = precedence
        elif record[1] == OperatorType.SUFFIX_UNARY:
            if up:
                raise exceptions.InvalidOperatorTableException(record[0])
            up = -precedence
        elif record[1] == OperatorType.BINARY_LEFT_ASSOCIATIVE:
            if bp:
                raise exceptions.InvalidOperatorTableException(record[0])
            bp = precedence
        elif record[1] == OperatorType.BINARY_RIGHT_ASSOCIATIVE:
            if bp:
                raise exceptions.InvalidOperatorTableException(record[0])
            bp = -precedence
        if record[0] == '[]':
            name = 'INDEXER'
        elif record[0] == '{}':
            name = 'MAP'
        else:
            name = name or 'OP_' + next(name_generator)
        operators[record[0]] = (up, bp, name, record[2] if len(record) > 2 else None)
    return YaqlOperators(operators, name_value_op)