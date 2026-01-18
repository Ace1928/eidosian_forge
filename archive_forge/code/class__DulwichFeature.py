import time
from io import BytesIO
from ... import errors as bzr_errors
from ... import tests
from ...tests.features import Feature, ModuleAvailableFeature
from .. import import_dulwich
class _DulwichFeature(Feature):

    def _probe(self):
        try:
            import_dulwich()
        except bzr_errors.DependencyNotPresent:
            return False
        return True

    def feature_name(self):
        return 'dulwich'