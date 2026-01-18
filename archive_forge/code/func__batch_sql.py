import multiprocessing
from typing import TYPE_CHECKING, Optional, Union
from .. import Dataset, Features, config
from ..formatting import query_table
from ..packaged_modules.sql.sql import Sql
from ..utils import tqdm as hf_tqdm
from .abc import AbstractDatasetInputStream
def _batch_sql(self, args):
    offset, index, to_sql_kwargs = args
    to_sql_kwargs = {**to_sql_kwargs, 'if_exists': 'append'} if offset > 0 else to_sql_kwargs
    batch = query_table(table=self.dataset.data, key=slice(offset, offset + self.batch_size), indices=self.dataset._indices)
    df = batch.to_pandas()
    num_rows = df.to_sql(self.name, self.con, index=index, **to_sql_kwargs)
    return num_rows or len(df)