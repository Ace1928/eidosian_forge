import os
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from filelock import BaseFileLock, Timeout
from . import config
from .arrow_dataset import Dataset
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager
from .features import Features
from .info import DatasetInfo, MetricInfo
from .naming import camelcase_to_snakecase
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
from .utils.py_utils import copyfunc, temp_seed
class MetricInfoMixin:
    """This base class exposes some attributes of MetricInfo
    at the base level of the Metric for easy access.

    <Deprecated version="2.5.0">

    Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>

    """

    def __init__(self, info: MetricInfo):
        self._metric_info = info

    @property
    def info(self):
        """:class:`datasets.MetricInfo` object containing all the metadata in the metric."""
        return self._metric_info

    @property
    def name(self) -> str:
        return self._metric_info.metric_name

    @property
    def experiment_id(self) -> Optional[str]:
        return self._metric_info.experiment_id

    @property
    def description(self) -> str:
        return self._metric_info.description

    @property
    def citation(self) -> str:
        return self._metric_info.citation

    @property
    def features(self) -> Features:
        return self._metric_info.features

    @property
    def inputs_description(self) -> str:
        return self._metric_info.inputs_description

    @property
    def homepage(self) -> Optional[str]:
        return self._metric_info.homepage

    @property
    def license(self) -> str:
        return self._metric_info.license

    @property
    def codebase_urls(self) -> Optional[List[str]]:
        return self._metric_info.codebase_urls

    @property
    def reference_urls(self) -> Optional[List[str]]:
        return self._metric_info.reference_urls

    @property
    def streamable(self) -> bool:
        return self._metric_info.streamable

    @property
    def format(self) -> Optional[str]:
        return self._metric_info.format