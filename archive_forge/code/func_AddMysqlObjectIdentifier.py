from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddMysqlObjectIdentifier(parser):
    """Adds a --mysql-database & --mysql-table flags to the given parser."""
    mysql_object_parser = parser.add_group()
    mysql_object_parser.add_argument('--mysql-database', help='Mysql database for the object.', required=True)
    mysql_object_parser.add_argument('--mysql-table', help='Mysql table for the object.', required=True)