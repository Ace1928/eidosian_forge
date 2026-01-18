from typing import Any, Optional
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame, ArrayDataFrame
from fugue.extensions.context import ExtensionContext
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
class OutputCoTransformer(CoTransformer):

    def process(self, dfs: DataFrames) -> None:
        raise NotImplementedError

    def get_output_schema(self, dfs: DataFrames) -> Any:
        return OUTPUT_TRANSFORMER_DUMMY_SCHEMA

    def transform(self, dfs: DataFrames) -> LocalDataFrame:
        self.process(dfs)
        return ArrayDataFrame([], OUTPUT_TRANSFORMER_DUMMY_SCHEMA)