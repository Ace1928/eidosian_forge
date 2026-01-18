import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def _get_valid_models(self):
    return ['model1', 'model2', 'model3']