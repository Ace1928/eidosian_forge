from unittest import mock
import oslotest.base as base
from osc_placement import version
def _convert_to_tuple(str):
    return tuple(map(int, str.split('.')))