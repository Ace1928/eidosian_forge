from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.assignment.role_backends import base
from keystone.assignment.role_backends import resource_options as ro
from keystone.common import resource_options
from keystone.common import sql
class ImpliedRoleTable(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'implied_role'
    attributes = ['prior_role_id', 'implied_role_id']
    prior_role_id = sql.Column(sql.String(64), sql.ForeignKey('role.id', ondelete='CASCADE'), primary_key=True)
    implied_role_id = sql.Column(sql.String(64), sql.ForeignKey('role.id', ondelete='CASCADE'), primary_key=True)

    @classmethod
    def from_dict(cls, dictionary):
        new_dictionary = dictionary.copy()
        return cls(**new_dictionary)

    def to_dict(self):
        """Return a dictionary with model's attributes.

        overrides the `to_dict` function from the base class
        to avoid having an `extra` field.
        """
        d = dict()
        for attr in self.__class__.attributes:
            d[attr] = getattr(self, attr)
        return d