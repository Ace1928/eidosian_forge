import warnings
from twisted.trial.unittest import TestCase
class STATUS(Values):
    OK = ValueConstant('200')
    NOT_FOUND = ValueConstant('404')