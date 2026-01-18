import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta
class non_deterministic:
    cls: Optional[Type[IterDataPipe]] = None
    deterministic_fn: Callable[[], bool]

    def __init__(self, arg: Union[Type[IterDataPipe], Callable[[], bool]]) -> None:
        if isinstance(arg, Type):
            if not issubclass(arg, IterDataPipe):
                raise TypeError(f'Only `IterDataPipe` can be decorated with `non_deterministic`, but {arg.__name__} is found')
            self.cls = arg
        elif isinstance(arg, Callable):
            self.deterministic_fn = arg
        else:
            raise TypeError(f'{arg} can not be decorated by non_deterministic')

    def __call__(self, *args, **kwargs):
        global _determinism
        if self.cls is not None:
            if _determinism:
                raise TypeError("{} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. You can turn off determinism for this DataPipe if that is acceptable for your application".format(self.cls.__name__))
            return self.cls(*args, **kwargs)
        if not (isinstance(args[0], Type) and issubclass(args[0], IterDataPipe)):
            raise TypeError(f'Only `IterDataPipe` can be decorated, but {args[0].__name__} is found')
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        res = self.deterministic_fn(*args, **kwargs)
        if not isinstance(res, bool):
            raise TypeError(f'deterministic_fn of `non_deterministic` decorator is required to return a boolean value, but {type(res)} is found')
        global _determinism
        if _determinism and res:
            raise TypeError(f"{self.cls.__name__} is non-deterministic with the inputs, but you set 'guaranteed_datapipes_determinism'. You can turn off determinism for this DataPipe if that is acceptable for your application")
        return self.cls(*args, **kwargs)