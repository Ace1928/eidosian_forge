from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBReferenceError(DBError):
    """Foreign key violation error.

    :param table: a table name in which the reference is directed.
    :type table: str
    :param constraint: a problematic constraint name.
    :type constraint: str
    :param key: a broken reference key name.
    :type key: str
    :param key_table: a table name which contains the key.
    :type key_table: str
    """

    def __init__(self, table, constraint, key, key_table, inner_exception=None):
        self.table = table
        self.constraint = constraint
        self.key = key
        self.key_table = key_table
        super(DBReferenceError, self).__init__(inner_exception)