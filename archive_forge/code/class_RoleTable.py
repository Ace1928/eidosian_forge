from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.assignment.role_backends import base
from keystone.assignment.role_backends import resource_options as ro
from keystone.common import resource_options
from keystone.common import sql
class RoleTable(sql.ModelBase, sql.ModelDictMixinWithExtras):

    def to_dict(self, include_extra_dict=False):
        d = super(RoleTable, self).to_dict(include_extra_dict=include_extra_dict)
        if d['domain_id'] == base.NULL_DOMAIN_ID:
            d['domain_id'] = None
        d['options'] = resource_options.ref_mapper_to_dict_options(self)
        return d

    @classmethod
    def from_dict(cls, role_dict):
        if 'domain_id' in role_dict and role_dict['domain_id'] is None:
            new_dict = role_dict.copy()
            new_dict['domain_id'] = base.NULL_DOMAIN_ID
        else:
            new_dict = role_dict
        resource_options = {}
        options = new_dict.pop('options', {})
        for opt in cls.resource_options_registry.options:
            if opt.option_name in options:
                opt_value = options[opt.option_name]
                if opt_value is not None:
                    opt.validator(opt_value)
                resource_options[opt.option_id] = opt_value
        role_obj = super(RoleTable, cls).from_dict(new_dict)
        setattr(role_obj, '_resource_options', resource_options)
        return role_obj
    __tablename__ = 'role'
    attributes = ['id', 'name', 'domain_id', 'description']
    resource_options_registry = ro.ROLE_OPTIONS_REGISTRY
    id = sql.Column(sql.String(64), primary_key=True)
    name = sql.Column(sql.String(255), nullable=False)
    domain_id = sql.Column(sql.String(64), nullable=False, server_default=base.NULL_DOMAIN_ID)
    description = sql.Column(sql.String(255), nullable=True)
    extra = sql.Column(sql.JsonBlob())
    _resource_option_mapper = orm.relationship('RoleOption', single_parent=True, cascade='all,delete,delete-orphan', lazy='subquery', backref='role', collection_class=collections.attribute_mapped_collection('option_id'))
    __table_args__ = (sql.UniqueConstraint('name', 'domain_id'),)