import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
class ReconnectMixin(object):
    """
    Mixin class that attempts to automatically reconnect to the database under
    certain error conditions.

    For example, MySQL servers will typically close connections that are idle
    for 28800 seconds ("wait_timeout" setting). If your application makes use
    of long-lived connections, you may find your connections are closed after
    a period of no activity. This mixin will attempt to reconnect automatically
    when these errors occur.

    This mixin class probably should not be used with Postgres (unless you
    REALLY know what you are doing) and definitely has no business being used
    with Sqlite. If you wish to use with Postgres, you will need to adapt the
    `reconnect_errors` attribute to something appropriate for Postgres.
    """
    reconnect_errors = ((OperationalError, '2006'), (OperationalError, '2013'), (OperationalError, '2014'), (OperationalError, '4031'), (OperationalError, 'MySQL Connection not available.'))

    def __init__(self, *args, **kwargs):
        super(ReconnectMixin, self).__init__(*args, **kwargs)
        self._reconnect_errors = {}
        for exc_class, err_fragment in self.reconnect_errors:
            self._reconnect_errors.setdefault(exc_class, [])
            self._reconnect_errors[exc_class].append(err_fragment.lower())

    def execute_sql(self, sql, params=None, commit=None):
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        return self._reconnect(super(ReconnectMixin, self).execute_sql, sql, params)

    def begin(self):
        return self._reconnect(super(ReconnectMixin, self).begin)

    def _reconnect(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if self.in_transaction():
                raise exc
            exc_class = type(exc)
            if exc_class not in self._reconnect_errors:
                raise exc
            exc_repr = str(exc).lower()
            for err_fragment in self._reconnect_errors[exc_class]:
                if err_fragment in exc_repr:
                    break
            else:
                raise exc
            if not self.is_closed():
                self.close()
                self.connect()
            return func(*args, **kwargs)