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
class DatabaseResource(testresources.TestResourceManager):
    """Database resource which connects and disconnects to a URL.

    For SQLite, this means the database is created implicitly, as a result
    of SQLite's usual behavior.  If the database is a file-based URL,
    it will remain after the resource has been torn down.

    For all other kinds of databases, the resource indicates to connect
    and disconnect from that database.

    """

    def __init__(self, database_type, _enginefacade=None, provision_new_database=True, ad_hoc_url=None):
        super(DatabaseResource, self).__init__()
        self.database_type = database_type
        self.provision_new_database = provision_new_database
        if _enginefacade:
            self._enginefacade = _enginefacade
        else:
            self._enginefacade = enginefacade._context_manager
        self.resources = [('backend', BackendResource(database_type, ad_hoc_url))]

    def make(self, dependency_resources):
        backend = dependency_resources['backend']
        _enginefacade = self._enginefacade.make_new_manager()
        if self.provision_new_database:
            db_token = _random_ident()
            url = backend.provisioned_database_url(db_token)
            LOG.info('CREATE BACKEND %s TOKEN %s', backend.engine.url, db_token)
            backend.create_named_database(db_token, conditional=True)
        else:
            db_token = None
            url = backend.url
        _enginefacade.configure(logging_name='%s@%s' % (self.database_type, db_token))
        _enginefacade._factory._start(connection=url)
        engine = _enginefacade._factory._writer_engine
        return ProvisionedDatabase(backend, _enginefacade, engine, db_token)

    def clean(self, resource):
        if self.provision_new_database:
            LOG.info('DROP BACKEND %s TOKEN %s', resource.backend.engine, resource.db_token)
            resource.backend.drop_named_database(resource.db_token)

    def isDirty(self):
        return False