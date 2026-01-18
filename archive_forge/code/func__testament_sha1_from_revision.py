from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
def _testament_sha1_from_revision(self, repository, revision_id):
    testament = StrictTestament3.from_revision(repository, revision_id)
    return testament.as_sha1()