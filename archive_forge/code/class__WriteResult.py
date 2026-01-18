from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
class _WriteResult:
    """Base class for write result classes."""
    __slots__ = ('__acknowledged',)

    def __init__(self, acknowledged: bool) -> None:
        self.__acknowledged = acknowledged

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__acknowledged})'

    def _raise_if_unacknowledged(self, property_name: str) -> None:
        """Raise an exception on property access if unacknowledged."""
        if not self.__acknowledged:
            raise InvalidOperation(f'A value for {property_name} is not available when the write is unacknowledged. Check the acknowledged attribute to avoid this error.')

    @property
    def acknowledged(self) -> bool:
        """Is this the result of an acknowledged write operation?

        The :attr:`acknowledged` attribute will be ``False`` when using
        ``WriteConcern(w=0)``, otherwise ``True``.

        .. note::
          If the :attr:`acknowledged` attribute is ``False`` all other
          attributes of this class will raise
          :class:`~pymongo.errors.InvalidOperation` when accessed. Values for
          other attributes cannot be determined if the write operation was
          unacknowledged.

        .. seealso::
          :class:`~pymongo.write_concern.WriteConcern`
        """
        return self.__acknowledged