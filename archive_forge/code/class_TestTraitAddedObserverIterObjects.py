import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
class TestTraitAddedObserverIterObjects(unittest.TestCase):
    """ Test iter_objects yields nothing. """

    def test_iter_objects_yields_nothing(self):
        observer = create_observer()
        actual = list(observer.iter_objects(None))
        self.assertEqual(actual, [])