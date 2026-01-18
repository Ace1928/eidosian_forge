import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class FieldsDict(MutableMapping):

    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, item):
        return self.inputs[item].value

    def __setitem__(self, item, value):
        self.inputs[item].value = value

    def __delitem__(self, item):
        raise KeyError('You cannot remove keys from ElementDict')

    def keys(self):
        return self.inputs.keys()

    def __contains__(self, item):
        return item in self.inputs

    def __iter__(self):
        return iter(self.inputs.keys())

    def __len__(self):
        return len(self.inputs)

    def __repr__(self):
        return '<%s for form %s>' % (self.__class__.__name__, self.inputs.form._name())