import unittest
from zope.interface.common import ABCInterface
from zope.interface.common import ABCInterfaceClass
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
def add_abc_interface_tests(cls, module):

    def predicate(iface):
        return iface.__module__ == module
    add_verify_tests(cls, iter_abc_interfaces(predicate))