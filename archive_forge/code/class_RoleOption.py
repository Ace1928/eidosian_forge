from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.assignment.role_backends import base
from keystone.assignment.role_backends import resource_options as ro
from keystone.common import resource_options
from keystone.common import sql
class RoleOption(sql.ModelBase):
    __tablename__ = 'role_option'
    role_id = sql.Column(sql.String(64), sql.ForeignKey('role.id', ondelete='CASCADE'), nullable=False, primary_key=True)
    option_id = sql.Column(sql.String(4), nullable=False, primary_key=True)
    option_value = sql.Column(sql.JsonBlob, nullable=True)

    def __init__(self, option_id, option_value):
        self.option_id = option_id
        self.option_value = option_value