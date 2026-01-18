import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class WithInput(nib.BaseInterface):

    class input_spec(nib.TraitedSpec):
        foo = nib.traits.Int(3, usedefault=True, max_ver='0.5')
    _version = '0.4'

    def _run_interface(self, runtime):
        return runtime