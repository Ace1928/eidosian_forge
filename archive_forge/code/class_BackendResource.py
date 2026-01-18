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
class BackendResource(testresources.TestResourceManager):

    def __init__(self, database_type, ad_hoc_url=None):
        super(BackendResource, self).__init__()
        self.database_type = database_type
        self.backend = Backend.backend_for_database_type(self.database_type)
        self.ad_hoc_url = ad_hoc_url
        if ad_hoc_url is None:
            self.backend = Backend.backend_for_database_type(self.database_type)
        else:
            self.backend = Backend(self.database_type, ad_hoc_url)
            self.backend._verify()

    def make(self, dependency_resources):
        return self.backend

    def clean(self, resource):
        self.backend._dispose()

    def isDirty(self):
        return False