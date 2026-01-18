import unittest
from traits.api import (
from traits.observation.api import (
class Student(HasTraits):
    """ Model for testing list + post_init (enthought/traits#275) """
    graduate = Event()