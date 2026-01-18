from oslo_utils import timeutils
import sqlalchemy as sa
from sqlalchemy import event  # noqa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import attributes
from sqlalchemy.orm import session as se
from neutron_lib._i18n import _
from neutron_lib.db import constants as db_const
from neutron_lib.db import model_base
from neutron_lib.db import sqlalchemytypes
class HasStandardAttributes(object):

    @classmethod
    def get_api_collections(cls):
        """Define the API collection this object will appear under.

        This should return a list of API collections that the object
        will be exposed under. Most should be exposed in just one
        collection (e.g. the network model is just exposed under
        'networks').

        This is used by the standard attr extensions to discover which
        resources need to be extended with the standard attr fields
        (e.g. created_at/updated_at/etc).
        """
        if hasattr(cls, 'api_collections'):
            return cls.api_collections
        raise NotImplementedError(_('%s must define api_collections') % cls)

    @classmethod
    def get_api_sub_resources(cls):
        """Define the API sub-resources this object will appear under.

        This should return a list of API sub-resources that the object
        will be exposed under.

        This is used by the standard attr extensions to discover which
        sub-resources need to be extended with the standard attr fields
        (e.g. created_at/updated_at/etc).
        """
        try:
            return cls.api_sub_resources
        except AttributeError:
            return []

    @classmethod
    def get_collection_resource_map(cls):
        try:
            return cls.collection_resource_map
        except AttributeError as e:
            raise NotImplementedError(_('%s must define collection_resource_map') % cls) from e

    @classmethod
    def validate_tag_support(cls):
        return getattr(cls, 'tag_support', False)

    @declarative.declared_attr
    def standard_attr_id(cls):
        return sa.Column(sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), sa.ForeignKey(StandardAttribute.id, ondelete='CASCADE'), unique=True, nullable=False)

    @declarative.declared_attr
    def standard_attr(cls):
        return sa.orm.relationship(StandardAttribute, lazy='joined', cascade='all, delete-orphan', single_parent=True, uselist=False)

    @property
    def _effective_standard_attribute_id(self):
        return self.standard_attr_id

    def __init__(self, *args, **kwargs):
        standard_attr_keys = ['description', 'created_at', 'updated_at', 'revision_number']
        standard_attr_kwargs = {}
        for key in standard_attr_keys:
            if key in kwargs:
                standard_attr_kwargs[key] = kwargs.pop(key)
        super().__init__(*args, **kwargs)
        self.standard_attr = StandardAttribute(resource_type=self.__tablename__, **standard_attr_kwargs)

    @declarative.declared_attr
    def description(cls):
        return association_proxy('standard_attr', 'description')

    @declarative.declared_attr
    def created_at(cls):
        return association_proxy('standard_attr', 'created_at')

    @declarative.declared_attr
    def updated_at(cls):
        return association_proxy('standard_attr', 'updated_at')

    def update(self, new_dict):
        new_dict.pop('created_at', None)
        new_dict.pop('updated_at', None)
        super().update(new_dict)

    @declarative.declared_attr
    def revision_number(cls):
        return association_proxy('standard_attr', 'revision_number')

    def bump_revision(self):
        self.standard_attr.bump_revision()

    def _set_updated_revision_number(self, revision_number, updated_at):
        self.standard_attr._set_updated_revision_number(revision_number, updated_at)