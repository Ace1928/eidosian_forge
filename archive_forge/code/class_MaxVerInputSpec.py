import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
class MaxVerInputSpec(nib.TraitedSpec):
    foo = nib.traits.Int(desc='a random int', max_ver='0.7')