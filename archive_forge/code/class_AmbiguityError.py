from django.db import DatabaseError
class AmbiguityError(Exception):
    """More than one migration matches a name prefix."""
    pass