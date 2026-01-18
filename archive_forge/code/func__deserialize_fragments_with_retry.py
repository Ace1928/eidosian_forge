import logging
from typing import (
import numpy as np
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.util import _check_pyarrow_version, _is_local_scheme
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource import Datasource
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import ReadTask
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.partitioning import PathPartitionFilter
from ray.data.datasource.path_util import (
from ray.util.annotations import PublicAPI
def _deserialize_fragments_with_retry(serialized_fragments: List[_SerializedFragment]) -> List['pyarrow._dataset.ParquetFileFragment']:
    """
    Deserialize the given serialized_fragments with retry upon errors.

    This retry helps when the upstream datasource is not able to handle
    overloaded read request or failed with some retriable failures.
    For example when reading data from HA hdfs service, hdfs might
    lose connection for some unknown reason expecially when
    simutaneously running many hyper parameter tuning jobs
    with ray.data parallelism setting at high value like the default 200
    Such connection failure can be restored with some waiting and retry.
    """
    min_interval = 0
    final_exception = None
    for i in range(FILE_READING_RETRY):
        try:
            return _deserialize_fragments(serialized_fragments)
        except Exception as e:
            import random
            import time
            retry_timing = '' if i == FILE_READING_RETRY - 1 else f'Retry after {min_interval} sec. '
            log_only_show_in_1st_retry = '' if i else f'If earlier read attempt threw certain Exception, it may or may not be an issue depends on these retries succeed or not. serialized_fragments:{serialized_fragments}'
            logger.exception(f'{i + 1}th attempt to deserialize ParquetFileFragment failed. {retry_timing}{log_only_show_in_1st_retry}')
            if not min_interval:
                min_interval = 1 + random.random()
            time.sleep(min_interval)
            min_interval = min_interval * 2
            final_exception = e
    raise final_exception