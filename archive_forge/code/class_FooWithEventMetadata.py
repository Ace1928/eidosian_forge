import unittest
from traits.api import (
from traits.observation.api import (
class FooWithEventMetadata(HasTraits):
    val = Str(event='the_trait')

    @observe('the_trait')
    def _handle_the_trait_changed(self, event):
        pass