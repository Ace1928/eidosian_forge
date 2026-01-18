from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
def _testament_sha1(self, revision_id):
    return StrictTestament3.from_revision(self.source, revision_id).as_sha1()