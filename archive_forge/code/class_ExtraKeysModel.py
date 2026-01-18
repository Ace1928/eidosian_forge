from collections import abc
import datetime
from unittest import mock
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import event
from sqlalchemy.orm import declarative_base
from oslo_db.sqlalchemy import models
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class ExtraKeysModel(BASE, models.ModelBase):
    __tablename__ = 'test_model'
    id = Column(Integer, primary_key=True)
    smth = Column(String(255))

    @property
    def name(self):
        return 'NAME'

    @property
    def _extra_keys(self):
        return ['name']