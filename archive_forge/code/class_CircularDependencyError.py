from django.db import DatabaseError
class CircularDependencyError(Exception):
    """There's an impossible-to-resolve circular dependency."""
    pass