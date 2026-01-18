import sys
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Optional
import pyarrow as pa
from .. import config
from ..features import Features
from ..features.features import decode_nested_example
from ..utils.py_utils import no_op_if_value_is_null
from .formatting import BaseArrowExtractor, TensorFormatter
class PolarsFormatter(TensorFormatter[Mapping, 'pl.DataFrame', Mapping]):

    def __init__(self, features=None, **np_array_kwargs):
        super().__init__(features=features)
        self.np_array_kwargs = np_array_kwargs
        self.polars_arrow_extractor = PolarsArrowExtractor
        self.polars_features_decoder = PolarsFeaturesDecoder(features)
        import polars as pl

    def format_row(self, pa_table: pa.Table) -> 'pl.DataFrame':
        row = self.polars_arrow_extractor().extract_row(pa_table)
        row = self.polars_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> 'pl.Series':
        column = self.polars_arrow_extractor().extract_column(pa_table)
        column = self.polars_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> 'pl.DataFrame':
        row = self.polars_arrow_extractor().extract_batch(pa_table)
        row = self.polars_features_decoder.decode_batch(row)
        return row