import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
class LimitModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'limit'
    attributes = ['internal_id', 'id', 'project_id', 'domain_id', 'service_id', 'region_id', 'resource_name', 'resource_limit', 'description', 'registered_limit_id']
    internal_id = sql.Column(sql.Integer, primary_key=True, nullable=False)
    id = sql.Column(sql.String(length=64), nullable=False, unique=True)
    project_id = sql.Column(sql.String(64))
    domain_id = sql.Column(sql.String(64))
    resource_limit = sql.Column(sql.Integer, nullable=False)
    description = sql.Column(sql.Text())
    registered_limit_id = sql.Column(sql.String(64), sql.ForeignKey('registered_limit.id'))
    registered_limit = sqlalchemy.orm.relationship('RegisteredLimitModel')

    @hybrid_property
    def service_id(self):
        if self.registered_limit:
            return self.registered_limit.service_id
        return None

    @service_id.expression
    def service_id(self):
        return RegisteredLimitModel.service_id

    @hybrid_property
    def region_id(self):
        if self.registered_limit:
            return self.registered_limit.region_id
        return None

    @region_id.expression
    def region_id(self):
        return RegisteredLimitModel.region_id

    @hybrid_property
    def resource_name(self):
        if self.registered_limit:
            return self.registered_limit.resource_name
        return self._resource_name

    @resource_name.expression
    def resource_name(self):
        return RegisteredLimitModel.resource_name

    def to_dict(self):
        ref = super(LimitModel, self).to_dict()
        if self.registered_limit:
            ref['service_id'] = self.registered_limit.service_id
            ref['region_id'] = self.registered_limit.region_id
            ref['resource_name'] = self.registered_limit.resource_name
        ref.pop('internal_id')
        ref.pop('registered_limit_id')
        return ref