import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def _convert_to_prover9(expression):
    """
    Convert ``logic.Expression`` to Prover9 formatted string.
    """
    if isinstance(expression, ExistsExpression):
        return 'exists ' + str(expression.variable) + ' ' + _convert_to_prover9(expression.term)
    elif isinstance(expression, AllExpression):
        return 'all ' + str(expression.variable) + ' ' + _convert_to_prover9(expression.term)
    elif isinstance(expression, NegatedExpression):
        return '-(' + _convert_to_prover9(expression.term) + ')'
    elif isinstance(expression, AndExpression):
        return '(' + _convert_to_prover9(expression.first) + ' & ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, OrExpression):
        return '(' + _convert_to_prover9(expression.first) + ' | ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, ImpExpression):
        return '(' + _convert_to_prover9(expression.first) + ' -> ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, IffExpression):
        return '(' + _convert_to_prover9(expression.first) + ' <-> ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, EqualityExpression):
        return '(' + _convert_to_prover9(expression.first) + ' = ' + _convert_to_prover9(expression.second) + ')'
    else:
        return str(expression)