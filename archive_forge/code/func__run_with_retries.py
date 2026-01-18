import time
from sqlalchemy import event
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from typing import Union, Optional
def _run_with_retries(fn, context, cursor_obj, statement, *arg, **kw):
    for retry in range(num_retries + 1):
        try:
            fn(cursor_obj, statement, *arg, context=context)
        except engine.dialect.dbapi.Error as raw_dbapi_err:
            connection = context.root_connection
            if engine.dialect.is_disconnect(raw_dbapi_err, connection, cursor_obj):
                if retry > num_retries:
                    raise
                engine.logger.error('disconnection error, retrying operation', exc_info=True)
                connection.invalidate()
                if hasattr(connection, 'rollback'):
                    connection.rollback()
                else:
                    trans = connection.get_transaction()
                    if trans:
                        trans.rollback()
                time.sleep(retry_interval)
                context.cursor = cursor_obj = connection.connection.cursor()
            else:
                raise
        else:
            return True