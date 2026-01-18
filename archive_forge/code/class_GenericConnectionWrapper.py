from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
class GenericConnectionWrapper:

    def __init__(self, baseconn):
        self._base = baseconn

    def __enter__(self):
        return self._base.__enter__()

    def __exit__(self, exc, value, tb):
        return self._base.__exit__(exc, value, tb)

    def __repr__(self):
        return self._base.__repr__()
    _proxy_funcs = ('affected_rows', 'autocommit', 'begin', 'change_user', 'character_set_name', 'close', 'commit', 'cursor', 'dump_debug_info', 'errno', 'error', 'errorhandler', 'get_server_info', 'insert_id', 'literal', 'ping', 'query', 'rollback', 'select_db', 'server_capabilities', 'set_character_set', 'set_isolation_level', 'set_server_option', 'set_sql_mode', 'show_warnings', 'shutdown', 'sqlstate', 'stat', 'store_result', 'string_literal', 'thread_id', 'use_result', 'warning_count')