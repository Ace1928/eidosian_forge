from typing import Any, Optional
class ExtensionError(SphinxError):
    """Extension error."""

    def __init__(self, message: str, orig_exc: Optional[Exception]=None, modname: Optional[str]=None) -> None:
        super().__init__(message)
        self.message = message
        self.orig_exc = orig_exc
        self.modname = modname

    @property
    def category(self) -> str:
        if self.modname:
            return 'Extension error (%s)' % self.modname
        else:
            return 'Extension error'

    def __repr__(self) -> str:
        if self.orig_exc:
            return '%s(%r, %r)' % (self.__class__.__name__, self.message, self.orig_exc)
        return '%s(%r)' % (self.__class__.__name__, self.message)

    def __str__(self) -> str:
        parent_str = super().__str__()
        if self.orig_exc:
            return '%s (exception: %s)' % (parent_str, self.orig_exc)
        return parent_str