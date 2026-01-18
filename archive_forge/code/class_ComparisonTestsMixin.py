from io import BytesIO
from xml.dom import minidom as dom
from twisted.internet.protocol import FileWrapper
class ComparisonTestsMixin:
    """
    A mixin which defines a method for making assertions about the correctness
    of an implementation of C{==} and C{!=}.

    Use this to unit test objects which follow the common convention for C{==}
    and C{!=}:

        - The object compares equal to itself
        - The object cooperates with unrecognized types to allow them to
          implement the comparison
        - The object implements not-equal as the opposite of equal
    """

    def assertNormalEqualityImplementation(self, firstValueOne, secondValueOne, valueTwo):
        """
        Assert that C{firstValueOne} is equal to C{secondValueOne} but not
        equal to C{valueOne} and that it defines equality cooperatively with
        other types it doesn't know about.

        @param firstValueOne: An object which is expected to compare as equal
            to C{secondValueOne} and not equal to C{valueTwo}.

        @param secondValueOne: A different object than C{firstValueOne} but
            which is expected to compare equal to that object.

        @param valueTwo: An object which is expected to compare as not equal to
            C{firstValueOne}.
        """
        self.assertTrue(firstValueOne == firstValueOne)
        self.assertTrue(firstValueOne == secondValueOne)
        self.assertFalse(firstValueOne == valueTwo)
        self.assertFalse(firstValueOne != firstValueOne)
        self.assertFalse(firstValueOne != secondValueOne)
        self.assertTrue(firstValueOne != valueTwo)
        self.assertTrue(firstValueOne == _Equal())
        self.assertFalse(firstValueOne != _Equal())
        self.assertFalse(firstValueOne == _NotEqual())
        self.assertTrue(firstValueOne != _NotEqual())