import warnings
from twisted.trial.unittest import TestCase
class ValuedLetters(Values):
    """
    Some more letters, with corresponding unicode values.
    """
    alpha = ValueConstant('Α')
    digamma = ValueConstant('Ϝ')
    zeta = ValueConstant('Ζ')