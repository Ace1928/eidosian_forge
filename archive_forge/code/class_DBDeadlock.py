from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBDeadlock(DBError):
    """Database dead lock error.

    Deadlock is a situation that occurs when two or more different database
    sessions have some data locked, and each database session requests a lock
    on the data that another, different, session has already locked.
    """

    def __init__(self, inner_exception=None):
        super(DBDeadlock, self).__init__(inner_exception)