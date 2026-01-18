import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
@classmethod
def _ensure_backend_available(cls, url):
    url = utils.make_url(url)
    try:
        eng = sqlalchemy.create_engine(url)
    except ImportError as i_e:
        LOG.info('The %(dbapi)s backend is unavailable: %(err)s', dict(dbapi=url.drivername, err=i_e))
        raise exception.BackendNotAvailable("Backend '%s' is unavailable: No DBAPI installed" % url.drivername)
    else:
        try:
            conn = eng.connect()
        except sqlalchemy.exc.DBAPIError as d_e:
            LOG.info('The %(dbapi)s backend is unavailable: %(err)s', dict(dbapi=url.drivername, err=d_e))
            raise exception.BackendNotAvailable("Backend '%s' is unavailable: Could not connect" % url.drivername)
        else:
            conn.close()
            return eng