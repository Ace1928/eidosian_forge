from typing import TYPE_CHECKING, Any, Callable, TypeVar
from typing_extensions import Annotated
from .. import _singleton
class _InnerMarker(_Marker):

    def __repr__(self):
        return description