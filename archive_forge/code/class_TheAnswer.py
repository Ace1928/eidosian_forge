from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
class TheAnswer(ValueWidget):
    _model_name = Unicode('TheAnswer')
    description = Unicode()

    def get_interact_value(self):
        return 42