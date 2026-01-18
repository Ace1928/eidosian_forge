import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import pyarrow as pa
import ray.data as rd
from pyarrow import csv as pacsv
from pyarrow import json as pajson
from ray.data.datasource import FileExtensionFilter
from triad.collections import Schema
from triad.collections.dict import ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import exists, makedirs, rm
from fugue import ExecutionEngine
from fugue._utils.io import FileParser, save_df
from fugue.collections.partition import PartitionSpec
from fugue.dataframe import DataFrame
from fugue_ray.dataframe import RayDataFrame
def _save_csv(self, df: RayDataFrame, uri: str, **kwargs: Any) -> None:
    kw = dict(kwargs)
    if 'header' in kw:
        kw['include_header'] = kw.pop('header')

    def _fn() -> Dict[str, Any]:
        return dict(write_options=pacsv.WriteOptions(**kw))
    df.native.write_csv(uri, ray_remote_args=self._remote_args(), arrow_csv_args_fn=_fn)