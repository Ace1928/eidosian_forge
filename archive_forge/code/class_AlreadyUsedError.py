from typing import Any, Dict
class AlreadyUsedError(RuntimeError):
    """An Outcome can only be unwrapped once."""
    pass