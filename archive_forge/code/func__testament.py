from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
def _testament(self, revision, tree):
    return StrictTestament3(revision, tree)