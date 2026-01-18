from typing import Any, Dict, Optional, Sequence
class UnsupportedDatabaseException(Exception):
    """Modin can't create a particular kind of database connection."""
    pass