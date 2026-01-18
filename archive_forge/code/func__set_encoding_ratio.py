import io
import logging
import time
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
import numpy as np
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.util import _check_import
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider
from ray.util.annotations import DeveloperAPI
def _set_encoding_ratio(self, encoding_ratio: int):
    """Set image file encoding ratio, to provide accurate size in bytes metadata."""
    self._encoding_ratio = encoding_ratio