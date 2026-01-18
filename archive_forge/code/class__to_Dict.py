from io import StringIO
import yaml
from unittest import mock
from cliff.formatters import yaml_format
from cliff.tests import base
from cliff.tests import test_columns
class _to_Dict:

    def __init__(self, **kwargs):
        self._data = kwargs

    def to_dict(self):
        return self._data