from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
class SomeImmutableMapping(abc.Mapping):

    def __init__(self):
        self.data = {'a': 2, 'b': 4, 'c': 8}

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)