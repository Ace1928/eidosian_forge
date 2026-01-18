import functools
from collections import namedtuple
from typing import Callable, Iterator, Sized, TypeVar, Optional, Union, Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (_check_unpickable_fn,
def _collate_helper(conversion, item):
    if len(item.items) > 1:
        raise Exception('Only supports one DataFrame per batch')
    df = item[0]
    columns_name = df_wrapper.get_columns(df)
    tuple_names: List = []
    tuple_values: List = []
    for name in conversion.keys():
        if name not in columns_name:
            raise Exception('Conversion keys missmatch')
    for name in columns_name:
        if name in conversion:
            if not callable(conversion[name]):
                raise Exception('Collate (DF)DataPipe requires callable as dict values')
            collation_fn = conversion[name]
        else:
            try:
                import torcharrow.pytorch as tap
                collation_fn = tap.rec.Default()
            except Exception as e:
                raise Exception('unable to import default collation function from the TorchArrow') from e
        tuple_names.append(str(name))
        value = collation_fn(df[name])
        tuple_values.append(value)
    tpl_cls = namedtuple('CollateResult', tuple_names)
    tuple = tpl_cls(*tuple_values)
    return tuple