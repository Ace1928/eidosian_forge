import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
from typing import Sequence as Sequence_
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, DatasetCard, DatasetCardData, HfApi
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .fingerprint import (
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.hub import list_files_info, preupload_lfs_files
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
def add_elasticsearch_index(self, column: str, index_name: Optional[str]=None, host: Optional[str]=None, port: Optional[int]=None, es_client: Optional['elasticsearch.Elasticsearch']=None, es_index_name: Optional[str]=None, es_index_config: Optional[dict]=None):
    """Add a text index using ElasticSearch for fast retrieval. This is done in-place.

        Args:
            column (`str`):
                The column of the documents to add to the index.
            index_name (`str`, *optional*):
                The `index_name`/identifier of the index.
                This is the index name that is used to call [`~Dataset.get_nearest_examples`] or [`Dataset.search`].
                By default it corresponds to `column`.
            host (`str`, *optional*, defaults to `localhost`):
                Host of where ElasticSearch is running.
            port (`str`, *optional*, defaults to `9200`):
                Port of where ElasticSearch is running.
            es_client (`elasticsearch.Elasticsearch`, *optional*):
                The elasticsearch client used to create the index if host and port are `None`.
            es_index_name (`str`, *optional*):
                The elasticsearch index name used to create the index.
            es_index_config (`dict`, *optional*):
                The configuration of the elasticsearch index.
                Default config is:
                    ```
                    {
                        "settings": {
                            "number_of_shards": 1,
                            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                        },
                        "mappings": {
                            "properties": {
                                "text": {
                                    "type": "text",
                                    "analyzer": "standard",
                                    "similarity": "BM25"
                                },
                            }
                        },
                    }
                    ```
        Example:

        ```python
        >>> es_client = elasticsearch.Elasticsearch()
        >>> ds = datasets.load_dataset('crime_and_punish', split='train')
        >>> ds.add_elasticsearch_index(column='line', es_client=es_client, es_index_name="my_es_index")
        >>> scores, retrieved_examples = ds.get_nearest_examples('line', 'my new query', k=10)
        ```
        """
    with self.formatted_as(type=None, columns=[column]):
        super().add_elasticsearch_index(column=column, index_name=index_name, host=host, port=port, es_client=es_client, es_index_name=es_index_name, es_index_config=es_index_config)
    return self