import unittest
from zope.interface.tests import OptimizationTestMixin
def _makeInterfaces():
    from zope.interface import Interface

    class IB0(Interface):
        pass

    class IB1(IB0):
        pass

    class IB2(IB0):
        pass

    class IB3(IB2, IB1):
        pass

    class IB4(IB1, IB2):
        pass

    class IF0(Interface):
        pass

    class IF1(IF0):
        pass

    class IR0(Interface):
        pass

    class IR1(IR0):
        pass
    return (IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1)