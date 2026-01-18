import functools
import itertools
import logging
import os
import re
import time
import debtcollector.removals
import debtcollector.renames
import sqlalchemy
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import pool
from sqlalchemy import select
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
def _vet_url(url):
    if '+' not in url.drivername and (not url.drivername.startswith('sqlite')):
        if url.drivername.startswith('mysql'):
            LOG.warning("URL %r does not contain a '+drivername' portion, and will make use of a default driver.  A full dbname+drivername:// protocol is recommended. For MySQL, it is strongly recommended that mysql+pymysql:// be specified for maximum service compatibility", url)
        else:
            LOG.warning("URL %r does not contain a '+drivername' portion, and will make use of a default driver.  A full dbname+drivername:// protocol is recommended.", url)