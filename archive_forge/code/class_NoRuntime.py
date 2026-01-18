import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class NoRuntime(TestInterface):

    def _run_interface(self, runtime):
        return None