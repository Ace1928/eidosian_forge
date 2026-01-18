import logging
import unittest
from oslo_utils import importutils
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import models
from oslo_db import tests
from oslo_db.tests.sqlalchemy import base as test_base
class TmpTable(BASE, models.ModelBase):
    __tablename__ = 'test_async_eventlet'
    id = sa.Column('id', sa.Integer, primary_key=True, nullable=False)
    foo = sa.Column('foo', sa.Integer)
    __table_args__ = (sa.UniqueConstraint('foo', name='uniq_foo'),)