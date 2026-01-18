from typing import TYPE_CHECKING
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import DeveloperAPI
def _read_subset(subset: 'torch.utils.data.Subset'):
    batch = []
    for item in subset:
        batch.append(item)
        if len(batch) == TORCH_DATASOURCE_READER_BATCH_SIZE:
            builder = DelegatingBlockBuilder()
            builder.add_batch({'item': batch})
            yield builder.build()
            batch.clear()
    if len(batch) > 0:
        builder = DelegatingBlockBuilder()
        builder.add_batch({'item': batch})
        yield builder.build()