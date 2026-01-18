import logging
import weakref
from threading import local as thread_local
from threading import Event
from threading import Thread
from peewee import __deprecated__
from playhouse.sqlite_ext import SqliteExtDatabase
class SqliteQueueDatabase(SqliteExtDatabase):
    WAL_MODE_ERROR_MESSAGE = 'SQLite must be configured to use the WAL journal mode when using this feature. WAL mode allows one or more readers to continue reading while another connection writes to the database.'

    def __init__(self, database, use_gevent=False, autostart=True, queue_max_size=None, results_timeout=None, *args, **kwargs):
        kwargs['check_same_thread'] = False
        pragmas = self._validate_journal_mode(kwargs.pop('pragmas', None))
        Parent = super(SqliteQueueDatabase, self)
        self._execute = Parent.execute_sql
        Parent.__init__(database, *args, pragmas=pragmas, **kwargs)
        self._autostart = autostart
        self._results_timeout = results_timeout
        self._is_stopped = True
        self._thread_helper = self.get_thread_impl(use_gevent)(queue_max_size)
        self._create_write_queue()
        if self._autostart:
            self.start()

    def get_thread_impl(self, use_gevent):
        return GreenletHelper if use_gevent else ThreadHelper

    def _validate_journal_mode(self, pragmas=None):
        if not pragmas:
            return {'journal_mode': 'wal'}
        if not isinstance(pragmas, dict):
            pragmas = dict(((k.lower(), v) for k, v in pragmas))
        if pragmas.get('journal_mode', 'wal').lower() != 'wal':
            raise ValueError(self.WAL_MODE_ERROR_MESSAGE)
        pragmas['journal_mode'] = 'wal'
        return pragmas

    def _create_write_queue(self):
        self._write_queue = self._thread_helper.queue()

    def queue_size(self):
        return self._write_queue.qsize()

    def execute_sql(self, sql, params=None, commit=None, timeout=None):
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        if sql.lower().startswith('select'):
            return self._execute(sql, params)
        cursor = AsyncCursor(event=self._thread_helper.event(), sql=sql, params=params, timeout=self._results_timeout if timeout is None else timeout)
        self._write_queue.put(cursor)
        return cursor

    def start(self):
        with self._lock:
            if not self._is_stopped:
                return False

            def run():
                writer = Writer(self, self._write_queue)
                writer.run()
            self._writer = self._thread_helper.thread(run)
            self._writer.start()
            self._is_stopped = False
            return True

    def stop(self):
        logger.debug('environment stop requested.')
        with self._lock:
            if self._is_stopped:
                return False
            self._write_queue.put(SHUTDOWN)
            self._writer.join()
            self._is_stopped = True
            return True

    def is_stopped(self):
        with self._lock:
            return self._is_stopped

    def pause(self):
        with self._lock:
            self._write_queue.put(PAUSE)

    def unpause(self):
        with self._lock:
            self._write_queue.put(UNPAUSE)

    def __unsupported__(self, *args, **kwargs):
        raise ValueError('This method is not supported by %r.' % type(self))
    atomic = transaction = savepoint = __unsupported__