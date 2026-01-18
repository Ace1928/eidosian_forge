from typing import Optional
class MissingOptionalLibraryError(QiskitError, ImportError):
    """Raised when an optional library is missing."""

    def __init__(self, libname: str, name: str, pip_install: Optional[str]=None, msg: Optional[str]=None) -> None:
        """Set the error message.
        Args:
            libname: Name of missing library
            name: Name of class, function, module that uses this library
            pip_install: pip install command, if any
            msg: Descriptive message, if any
        """
        message = [f"The '{libname}' library is required to use '{name}'."]
        if pip_install:
            message.append(f"You can install it with '{pip_install}'.")
        if msg:
            message.append(f' {msg}.')
        super().__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self) -> str:
        """Return the message."""
        return repr(self.message)