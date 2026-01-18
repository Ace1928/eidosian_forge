import os
from pathlib import Path
import unittest
from traits.api import File, HasTraits, TraitError
from traits.testing.optional_dependencies import requires_traitsui
class FastExampleModel(HasTraits):
    file_name = File()