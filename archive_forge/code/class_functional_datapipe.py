import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta
class functional_datapipe:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        """
        Define a functional datapipe.

        Args:
            enable_df_api_tracing - if set, any returned DataPipe would accept
            DataFrames API in tracing mode.
        """
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        if issubclass(cls, IterDataPipe):
            if isinstance(cls, Type):
                if not isinstance(cls, _DataPipeMeta):
                    raise TypeError('`functional_datapipe` can only decorate IterDataPipe')
            elif not isinstance(cls, non_deterministic) and (not (hasattr(cls, '__self__') and isinstance(cls.__self__, non_deterministic))):
                raise TypeError('`functional_datapipe` can only decorate IterDataPipe')
            IterDataPipe.register_datapipe_as_function(self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing)
        elif issubclass(cls, MapDataPipe):
            MapDataPipe.register_datapipe_as_function(self.name, cls)
        return cls