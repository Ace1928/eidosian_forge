from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
class BundleReaderV09(BundleReader):
    """BundleReader for 0.9 bundles"""

    def _get_info(self):
        return BundleInfo09()