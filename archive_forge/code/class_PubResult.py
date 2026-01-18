from __future__ import annotations
from typing import Any
from .data_bin import DataBin
class PubResult:
    """Result of Primitive Unified Bloc."""
    __slots__ = ('_data', '_metadata')

    def __init__(self, data: DataBin, metadata: dict[str, Any] | None=None):
        """Initialize a pub result.

        Args:
            data: Result data.
            metadata: Metadata specific to this pub. Keys are expected to be strings.
        """
        self._data = data
        self._metadata = metadata or {}

    def __repr__(self):
        metadata = f', metadata={self.metadata}' if self.metadata else ''
        return f'{type(self).__name__}(data={self._data}{metadata})'

    @property
    def data(self) -> DataBin:
        """Result data for the pub."""
        return self._data

    @property
    def metadata(self) -> dict:
        """Metadata for the pub."""
        return self._metadata