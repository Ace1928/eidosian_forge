import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
class MockTransform1(Transformer):

    def get_output_schema(self, df: DataFrame) -> Any:
        assert 'test' in self.workflow_conf
        return [df.schema, 'ct:int,p:int']

    def on_init(self, df: DataFrame) -> None:
        assert 'test' in self.workflow_conf
        self.pn = self.cursor.physical_partition_no
        self.ks = self.key_schema
        if 'on_init_called' not in self.__dict__:
            self.on_init_called = 1
        else:
            self.on_init_called += 1

    def transform(self, df: LocalDataFrame) -> LocalDataFrame:
        assert 1 == self.on_init_called
        assert 'test' in self.workflow_conf
        pdf = df.as_pandas()
        pdf['p'] = self.params.get('p', 1)
        pdf['ct'] = pdf.shape[0]
        return PandasDataFrame(pdf, self.output_schema)