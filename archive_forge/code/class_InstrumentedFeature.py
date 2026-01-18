import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
class InstrumentedFeature(features.Feature):

    def __init__(self):
        super().__init__()
        self.calls = []

    def _probe(self):
        self.calls.append('_probe')
        return False