from typing import Any, List, Generic, TypeVar, Optional, cast
from typing_extensions import Protocol, override, runtime_checkable
from ._base_client import BasePage, PageInfo, BaseSyncPage, BaseAsyncPage
@runtime_checkable
class CursorPageItem(Protocol):
    id: Optional[str]