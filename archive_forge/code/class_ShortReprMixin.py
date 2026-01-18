import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
class ShortReprMixin:

    def __repr__(self) -> str:
        clas = self.__class__.__name__
        props = {k: getattr(self, k) for k, v in self.__class__.__dict__.items() if isinstance(v, Attr)}
        settings = [f'{k}={v!r}' for k, v in props.items() if not self._is_interesting(v)]
        return '{}({})'.format(clas, ', '.join(settings))

    @staticmethod
    def _is_interesting(x: Any) -> bool:
        if isinstance(x, (list, tuple)):
            return all((v is None for v in x))
        return x is None or x == {}