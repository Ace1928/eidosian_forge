import sys
import unittest
class IHasAttribute(Interface):
    """ This interface has an attribute.
            """
    an_attribute = Attribute('an_attribute', 'This attribute is documented.')