import unittest
from traits.api import (
class ObjectWithReadOnlyText(HasTraits):
    """ A dummy object that set the readonly trait in __init__

    There exists such usage in TraitsUI.
    """
    text = ReadOnly()

    def __init__(self, text, **traits):
        self.text = text
        super(ObjectWithReadOnlyText, self).__init__(**traits)