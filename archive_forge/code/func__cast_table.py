import itertools
from dataclasses import dataclass
from typing import Optional
import pyarrow as pa
import datasets
from datasets.table import table_cast
def _cast_table(self, pa_table: pa.Table) -> pa.Table:
    if self.info.features is not None:
        pa_table = table_cast(pa_table, self.info.features.arrow_schema)
    return pa_table