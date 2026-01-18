import numbers as abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IRational(IReal):
    abc = abc.Rational