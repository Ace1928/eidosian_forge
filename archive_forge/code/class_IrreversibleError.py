from django.db import DatabaseError
class IrreversibleError(RuntimeError):
    """An irreversible migration is about to be reversed."""
    pass