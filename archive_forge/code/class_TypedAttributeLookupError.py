from __future__ import annotations
class TypedAttributeLookupError(LookupError):
    """
    Raised by :meth:`~anyio.TypedAttributeProvider.extra` when the given typed attribute
    is not found and no default value has been given.
    """