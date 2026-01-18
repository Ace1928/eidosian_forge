from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddDatabaseServiceFlag(parser):
    """Add the database service field to the parser."""
    parser.add_argument('--database-service', required=True, help='database service for the oracle connection profile.')