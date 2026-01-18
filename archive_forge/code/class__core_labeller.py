from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
class _core_labeller(metaclass=ABCMeta):
    """
    Per item
    """

    @abstractmethod
    def __call__(self, label_info: strip_label_details) -> strip_label_details:
        pass