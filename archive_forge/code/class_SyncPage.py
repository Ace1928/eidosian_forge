from typing import Any, List, Generic, TypeVar, Optional, cast
from typing_extensions import Protocol, override, runtime_checkable
from ._base_client import BasePage, PageInfo, BaseSyncPage, BaseAsyncPage
class SyncPage(BaseSyncPage[_T], BasePage[_T], Generic[_T]):
    """Note: no pagination actually occurs yet, this is for forwards-compatibility."""
    data: List[_T]
    object: str

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def next_page_info(self) -> None:
        """
        This page represents a response that isn't actually paginated at the API level
        so there will never be a next page.
        """
        return None