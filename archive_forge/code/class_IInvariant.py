import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class IInvariant(Interface):
    foo = Attribute('foo')
    bar = Attribute('bar; must eval to Boolean True if foo does')
    invariant(_ifFooThenBar)