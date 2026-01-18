import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
@dataclass
@Deprecated(DATASET_CONFIG_DEPRECATION_MSG)
class DatasetConfig:
    """Configuration for ingest of a single Dataset.

    See :ref:`the AIR Dataset configuration guide <data-ingest-torch>` for
    usage examples.

    This config defines how the Dataset should be read into the DataParallelTrainer.
    It configures the preprocessing, splitting, and ingest strategy per-dataset.

    DataParallelTrainers declare default DatasetConfigs for each dataset passed in the
    ``datasets`` argument. Users have the opportunity to selectively override these
    configs by passing the ``dataset_config`` argument. Trainers can also define user
    customizable values (e.g., XGBoostTrainer doesn't support streaming ingest).

    Args:
        fit: Whether to fit preprocessors on this dataset. This can be set on at most
            one dataset at a time. True by default for the "train" dataset only.
        split: Whether the dataset should be split across multiple workers.
            True by default for the "train" dataset only.
        required: Whether to raise an error if the Dataset isn't provided by the user.
            False by default.
        transform: Whether to transform the dataset with the fitted preprocessor.
            This must be enabled at least for the dataset that is fit.
            True by default.
        max_object_store_memory_fraction [Experimental]: The maximum fraction
            of Ray's shared-memory object store to use for the dataset. The
            default value is -1, meaning that the preprocessed dataset should
            be cached, which may cause spilling if its size is larger than the
            object store's capacity. Pipelined ingest (all other values, 0 or
            higher) is experimental. Note that the absolute memory capacity
            used is based on the object store capacity at invocation time; this
            does not currently cover autoscaling cases where the size of the
            cluster may change.
        global_shuffle: Whether to enable global shuffle (per pipeline window
            in streaming mode). Note that this is an expensive all-to-all operation,
            and most likely you want to use local shuffle instead.
            See https://docs.ray.io/en/master/data/faq.html and
            https://docs.ray.io/en/master/ray-air/check-ingest.html.
            False by default.
        randomize_block_order: Whether to randomize the iteration order over blocks.
            The main purpose of this is to prevent data fetching hotspots in the
            cluster when running many parallel workers / trials on the same data.
            We recommend enabling it always. True by default.
        per_epoch_preprocessor [Experimental]: A preprocessor to re-apply on
            each pass of the dataset. The main use case for this is to apply a
            random transform on a training dataset on each epoch. The
            per-epoch preprocessor will be applied *after* all other
            preprocessors and in parallel with the dataset consumer.
        use_stream_api: Deprecated. Use max_object_store_memory_fraction instead.
        stream_window_size: Deprecated. Use max_object_store_memory_fraction instead.
    """
    fit: Optional[bool] = None
    split: Optional[bool] = None
    required: Optional[bool] = None
    transform: Optional[bool] = None
    max_object_store_memory_fraction: Optional[float] = None
    global_shuffle: Optional[bool] = None
    randomize_block_order: Optional[bool] = None
    per_epoch_preprocessor: Optional['Preprocessor'] = None
    use_stream_api: Optional[int] = None
    stream_window_size: Optional[int] = None

    def __post_init__(self):
        raise DeprecationWarning(DATASET_CONFIG_DEPRECATION_MSG)