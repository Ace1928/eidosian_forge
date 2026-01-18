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
class ProvisionedDatabase(object):
    """Represents a database engine pointing to a DB ready to run tests.

    backend: an instance of :class:`.Backend`

    enginefacade: an instance of :class:`._TransactionFactory`

    engine: a SQLAlchemy :class:`.Engine`

    db_token: if provision_new_database were used, this is the randomly
              generated name of the database.  Note that with SQLite memory
              connections, this token is ignored.   For a database that
              wasn't actually created, will be None.

    """
    __slots__ = ('backend', 'enginefacade', 'engine', 'db_token')

    def __init__(self, backend, enginefacade, engine, db_token):
        self.backend = backend
        self.enginefacade = enginefacade
        self.engine = engine
        self.db_token = db_token