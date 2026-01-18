from typing import Any, List, Generic, TypeVar, Optional, cast
from typing_extensions import Protocol, override, runtime_checkable
from ._base_client import BasePage, PageInfo, BaseSyncPage, BaseAsyncPage
class SyncCursorPage(BaseSyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        data = self.data
        if not data:
            return None
        item = cast(Any, data[-1])
        if not isinstance(item, CursorPageItem) or item.id is None:
            return None
        return PageInfo(params={'after': item.id})