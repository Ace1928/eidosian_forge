from .... import errors as bzr_errors
from ....tests import TestLoader
from ....tests.features import Feature
from .. import load_fastimport
class _FastimportFeature(Feature):

    def _probe(self):
        try:
            load_fastimport()
        except bzr_errors.DependencyNotPresent:
            return False
        return True

    def feature_name(self):
        return 'fastimport'