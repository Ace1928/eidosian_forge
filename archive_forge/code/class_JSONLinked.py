import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
class JSONLinked(Property):
    """Property that is linked to one or more JSON keys."""

    def __init__(self, *args: Func, json_path: Optional[Union[str, List[str]]]=None, **kwargs: Func) -> None:
        super().__init__(*args, **kwargs)
        self.path_or_name = json_path

    def __set_name__(self, owner: Any, name: str) -> None:
        if self.path_or_name is None:
            self.path_or_name = name
        super().__set_name__(owner, name)

    def getter(self, fget: Func) -> Func:
        prop = type(self)(fget, self.fset, json_path=self.path_or_name)
        prop.name = self.name
        return prop

    def setter(self, fset: Func) -> Func:
        prop = type(self)(self.fget, fset, json_path=self.path_or_name)
        prop.name = self.name
        return prop

    def default_fget(self, obj: Any) -> Union[Any, List[Any]]:
        if isinstance(self.path_or_name, str):
            return nested_get(obj, self.path_or_name)
        elif isinstance(self.path_or_name, list):
            return [nested_get(obj, p) for p in self.path_or_name]
        else:
            raise TypeError(f'Unexpected type for path {type(self.path_or_name)!r}')

    def default_fset(self, obj: Any, value: Any) -> None:
        if isinstance(self.path_or_name, str):
            nested_set(obj, self.path_or_name, value)
        elif isinstance(self.path_or_name, list):
            for p, v in zip(self.path_or_name, value):
                nested_set(obj, p, v)
        else:
            raise TypeError(f'Unexpected type for path {type(self.path_or_name)!r}')