from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBConstraintError(DBError):
    """Check constraint fails for column error.

    Raised when made an attempt to write to a column a value that does not
    satisfy a CHECK constraint.

    :kwarg table: the table name for which the check fails
    :type table: str
    :kwarg check_name: the table of the check that failed to be satisfied
    :type check_name: str
    """

    def __init__(self, table, check_name, inner_exception=None):
        self.table = table
        self.check_name = check_name
        super(DBConstraintError, self).__init__(inner_exception)