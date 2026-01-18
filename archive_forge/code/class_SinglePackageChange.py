from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SinglePackageChange(_messages.Message):
    """Options to configure rule type SinglePackageChange. The rule is used to
  alter the sql code for a package entities. The rule filter field can refer
  to one entity. The rule scope can be: Package

  Fields:
    packageBody: Optional. Sql code for package body
    packageDescription: Optional. Sql code for package description
  """
    packageBody = _messages.StringField(1)
    packageDescription = _messages.StringField(2)