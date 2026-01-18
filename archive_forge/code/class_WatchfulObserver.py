import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
class WatchfulObserver(DummyObserver):
    """ This is a dummy observer to be used as the next observer following
    FilteredTraitObserver.
    """

    def iter_observables(self, object):
        if object in (Undefined, Uninitialized, None):
            raise ValueError('Child observer unexpectedly receive {}'.format(object))
        yield from self.observables