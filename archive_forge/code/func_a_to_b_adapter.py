import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def a_to_b_adapter(adaptee):
    if adaptee.allow_adaptation:
        b = B()
        b.marker = True
    else:
        b = None
    return b