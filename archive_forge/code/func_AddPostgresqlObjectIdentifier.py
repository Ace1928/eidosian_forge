from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddPostgresqlObjectIdentifier(parser):
    """Adds a --postgresql-schema & --postgresql-table flags to the given parser."""
    postgresql_object_parser = parser.add_group()
    postgresql_object_parser.add_argument('--postgresql-schema', help='PostgreSQL schema for the object.', required=True)
    postgresql_object_parser.add_argument('--postgresql-table', help='PostgreSQL table for the object.', required=True)