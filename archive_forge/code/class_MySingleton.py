from sympy.core.basic import Basic
from sympy.core.numbers import Rational
from sympy.core.singleton import S, Singleton
class MySingleton(Basic, metaclass=Singleton):
    pass