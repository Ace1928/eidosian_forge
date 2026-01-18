from django.db import DatabaseError
class MigrationSchemaMissing(DatabaseError):
    pass