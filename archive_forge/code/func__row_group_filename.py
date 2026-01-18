import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def _row_group_filename(self, row_group, metadata):
    raise NotImplementedError