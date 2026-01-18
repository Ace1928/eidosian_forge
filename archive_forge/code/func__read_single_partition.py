import logging
from typing import List, Optional
from ray.data._internal.util import _check_import
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _read_single_partition(stream) -> Block:
    client = bigquery_storage.BigQueryReadClient()
    reader = client.read_rows(stream.name)
    return reader.to_arrow()