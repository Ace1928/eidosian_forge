import json
from sqlalchemy.dialects import mysql
from sqlalchemy.types import Integer, Text, TypeDecorator
class SoftDeleteInteger(TypeDecorator):
    """Coerce a bound param to be a proper integer before passing it to DBAPI.

    Some backends like PostgreSQL are very strict about types and do not
    perform automatic type casts, e.g. when trying to INSERT a boolean value
    like ``false`` into an integer column. Coercing of the bound param in DB
    layer by the means of a custom SQLAlchemy type decorator makes sure we
    always pass a proper integer value to a DBAPI implementation.

    This is not a general purpose boolean integer type as it specifically
    allows for arbitrary positive integers outside of the boolean int range
    (0, 1, False, True), so that it's possible to have compound unique
    constraints over multiple columns including ``deleted`` (e.g. to
    soft-delete flavors with the same name in Nova without triggering
    a constraint violation): ``deleted`` is set to be equal to a PK
    int value on deletion, 0 denotes a non-deleted row.

    """
    impl = Integer
    cache_ok = True
    'This type is safe to cache.'

    def process_bind_param(self, value, dialect):
        """Return the binding parameter."""
        if value is None:
            return None
        return int(value)