import logging
from typing import Callable, Generic, List
from typing_extensions import ParamSpec  # Python 3.10+
class CallbackRegistry(Generic[P]):

    def __init__(self, name: str):
        self.name = name
        self.callback_list: List[Callable[P, None]] = []

    def add_callback(self, cb: Callable[P, None]) -> None:
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        for cb in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.exception('Exception in callback for %s registered with CUDA trace', self.name)