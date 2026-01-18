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
def _rows_per_file(self):
    return 1