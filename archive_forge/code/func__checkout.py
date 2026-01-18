from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
@classmethod
def _checkout(cls, pool: Pool, threadconns: Optional[threading.local]=None, fairy: Optional[_ConnectionFairy]=None) -> _ConnectionFairy:
    if not fairy:
        fairy = _ConnectionRecord.checkout(pool)
        if threadconns is not None:
            threadconns.current = weakref.ref(fairy)
    assert fairy._connection_record is not None, "can't 'checkout' a detached connection fairy"
    assert fairy.dbapi_connection is not None, "can't 'checkout' an invalidated connection fairy"
    fairy._counter += 1
    if not pool.dispatch.checkout and (not pool._pre_ping) or fairy._counter != 1:
        return fairy
    attempts = 2
    while attempts > 0:
        connection_is_fresh = fairy._connection_record.fresh
        fairy._connection_record.fresh = False
        try:
            if pool._pre_ping:
                if not connection_is_fresh:
                    if fairy._echo:
                        pool.logger.debug('Pool pre-ping on connection %s', fairy.dbapi_connection)
                    result = pool._dialect._do_ping_w_event(fairy.dbapi_connection)
                    if not result:
                        if fairy._echo:
                            pool.logger.debug('Pool pre-ping on connection %s failed, will invalidate pool', fairy.dbapi_connection)
                        raise exc.InvalidatePoolError()
                elif fairy._echo:
                    pool.logger.debug('Connection %s is fresh, skipping pre-ping', fairy.dbapi_connection)
            pool.dispatch.checkout(fairy.dbapi_connection, fairy._connection_record, fairy)
            return fairy
        except exc.DisconnectionError as e:
            if e.invalidate_pool:
                pool.logger.info('Disconnection detected on checkout, invalidating all pooled connections prior to current timestamp (reason: %r)', e)
                fairy._connection_record.invalidate(e)
                pool._invalidate(fairy, e, _checkin=False)
            else:
                pool.logger.info('Disconnection detected on checkout, invalidating individual connection %s (reason: %r)', fairy.dbapi_connection, e)
                fairy._connection_record.invalidate(e)
            try:
                fairy.dbapi_connection = fairy._connection_record.get_connection()
            except BaseException as err:
                with util.safe_reraise():
                    fairy._connection_record._checkin_failed(err, _fairy_was_created=True)
                    del fairy
                raise
            attempts -= 1
        except BaseException as be_outer:
            with util.safe_reraise():
                rec = fairy._connection_record
                if rec is not None:
                    rec._checkin_failed(be_outer, _fairy_was_created=True)
                del fairy
            raise
    pool.logger.info('Reconnection attempts exhausted on checkout')
    fairy.invalidate()
    raise exc.InvalidRequestError('This connection is closed')