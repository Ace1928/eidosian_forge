from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Set
from sqlalchemy import CHAR
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Numeric
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from ... import autogenerate
from ... import util
from ...autogenerate import api
from ...ddl.base import _fk_spec
from ...migration import MigrationContext
from ...operations import ops
from ...testing import config
from ...testing import eq_
from ...testing.env import clear_staging_env
from ...testing.env import staging_env
class ModelOne:
    __requires__ = ('unique_constraint_reflection',)
    schema: Any = None

    @classmethod
    def _get_db_schema(cls):
        schema = cls.schema
        m = MetaData(schema=schema)
        Table('user', m, Column('id', Integer, primary_key=True), Column('name', String(50)), Column('a1', Text), Column('pw', String(50)), Index('pw_idx', 'pw'))
        Table('address', m, Column('id', Integer, primary_key=True), Column('email_address', String(100), nullable=False))
        Table('order', m, Column('order_id', Integer, primary_key=True), Column('amount', Numeric(8, 2), nullable=False, server_default=text('0')), CheckConstraint('amount >= 0', name='ck_order_amount'))
        Table('extra', m, Column('x', CHAR), Column('uid', Integer, ForeignKey('user.id')))
        return m

    @classmethod
    def _get_model_schema(cls):
        schema = cls.schema
        m = MetaData(schema=schema)
        Table('user', m, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', Text, server_default='x'))
        Table('address', m, Column('id', Integer, primary_key=True), Column('email_address', String(100), nullable=False), Column('street', String(50)), UniqueConstraint('email_address', name='uq_email'))
        Table('order', m, Column('order_id', Integer, primary_key=True), Column('amount', Numeric(10, 2), nullable=True, server_default=text('0')), Column('user_id', Integer, ForeignKey('user.id')), CheckConstraint('amount > -1', name='ck_order_amount'))
        Table('item', m, Column('id', Integer, primary_key=True), Column('description', String(100)), Column('order_id', Integer, ForeignKey('order.order_id')), CheckConstraint('len(description) > 5'))
        return m