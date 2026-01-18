from typing import Any, Optional
class NoUri(Exception):
    """Raised by builder.get_relative_uri() or from missing-reference handlers
    if there is no URI available."""
    pass