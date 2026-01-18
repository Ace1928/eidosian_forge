import numbers as abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IIntegral(IRational):
    abc = abc.Integral