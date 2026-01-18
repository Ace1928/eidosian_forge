from __future__ import annotations
from typing import Any, Optional
class ReadConcern:
    """ReadConcern

    :Parameters:
        - `level`: (string) The read concern level specifies the level of
          isolation for read operations.  For example, a read operation using a
          read concern level of ``majority`` will only return data that has been
          written to a majority of nodes. If the level is left unspecified, the
          server default will be used.

    .. versionadded:: 3.2

    """

    def __init__(self, level: Optional[str]=None) -> None:
        if level is None or isinstance(level, str):
            self.__level = level
        else:
            raise TypeError('level must be a string or None.')

    @property
    def level(self) -> Optional[str]:
        """The read concern level."""
        return self.__level

    @property
    def ok_for_legacy(self) -> bool:
        """Return ``True`` if this read concern is compatible with
        old wire protocol versions.
        """
        return self.level is None or self.level == 'local'

    @property
    def document(self) -> dict[str, Any]:
        """The document representation of this read concern.

        .. note::
          :class:`ReadConcern` is immutable. Mutating the value of
          :attr:`document` does not mutate this :class:`ReadConcern`.
        """
        doc = {}
        if self.__level:
            doc['level'] = self.level
        return doc

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ReadConcern):
            return self.document == other.document
        return NotImplemented

    def __repr__(self) -> str:
        if self.level:
            return 'ReadConcern(%s)' % self.level
        return 'ReadConcern()'