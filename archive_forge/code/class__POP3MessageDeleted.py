from typing import Optional
class _POP3MessageDeleted(Exception):
    """
    An internal control-flow error which indicates that a deleted message was
    requested.
    """