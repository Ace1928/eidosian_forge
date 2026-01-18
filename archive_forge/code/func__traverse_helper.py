import io
import pickle
import warnings
from collections.abc import Collection
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def _traverse_helper(datapipe: DataPipe, only_datapipe: bool, cache: Set[int]) -> DataPipeGraph:
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise RuntimeError(f'Expected `IterDataPipe` or `MapDataPipe`, but {type(datapipe)} is found')
    dp_id = id(datapipe)
    if dp_id in cache:
        return {}
    cache.add(dp_id)
    items = _list_connected_datapipes(datapipe, only_datapipe, cache.copy())
    d: DataPipeGraph = {dp_id: (datapipe, {})}
    for item in items:
        d[dp_id][1].update(_traverse_helper(item, only_datapipe, cache.copy()))
    return d