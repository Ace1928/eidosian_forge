from typing import Any, Optional
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame, ArrayDataFrame
from fugue.extensions.context import ExtensionContext
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
class OutputTransformer(Transformer):

    def process(self, df: LocalDataFrame) -> None:
        raise NotImplementedError

    def get_output_schema(self, df: DataFrame) -> Any:
        return OUTPUT_TRANSFORMER_DUMMY_SCHEMA

    def transform(self, df: LocalDataFrame) -> LocalDataFrame:
        self.process(df)
        return ArrayDataFrame([], OUTPUT_TRANSFORMER_DUMMY_SCHEMA)