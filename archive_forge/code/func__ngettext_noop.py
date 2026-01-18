from __future__ import annotations
import gettext as gettext_module
import os.path
from threading import local
def _ngettext_noop(singular: str, plural: str) -> tuple[str, str]:
    """Mark two strings as pluralized translations without translating them.

    Example usage:
    ```python
    CONSTANTS = [ngettext_noop('first', 'firsts'), ngettext_noop('second', 'seconds')]
    def num_name(n):
        return _ngettext(*CONSTANTS[n])
    ```

    Args:
        singular (str): Singular text to translate in the future.
        plural (str): Plural text to translate in the future.

    Returns:
        tuple: Original text, unchanged.
    """
    return (singular, plural)