import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class DerivedInterface2(nib.BaseInterface):
    input_spec = MaxVerInputSpec
    _version = '0.8'