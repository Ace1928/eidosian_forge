from typing import Any, Optional
class SphinxWarning(SphinxError):
    """Warning, treated as error."""
    category = 'Warning, treated as error'