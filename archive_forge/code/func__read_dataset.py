import sys
from typing import TYPE_CHECKING, Iterable, List, Optional, Union
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.util.annotations import DeveloperAPI
def _read_dataset(dataset: 'datasets.IterableDataset') -> Iterable[Block]:
    for batch in dataset.with_format('arrow').iter(batch_size=self._batch_size):
        if not isinstance(batch, (pyarrow.Table, pd.DataFrame, dict, np.array)):
            raise ValueError(f"Batch format {type(batch)} isn't supported. Only the following batch formats are supported: dict (corresponds to `None` in `dataset.with_format()`), pyarrow.Table, np.array, pd.DataFrame.")
        if isinstance(batch, np.ndarray):
            batch = {'item': batch}
        if isinstance(batch, dict):
            batch = pyarrow.Table.from_pydict(batch)
        block = BlockAccessor.for_block(batch).to_default()
        yield block