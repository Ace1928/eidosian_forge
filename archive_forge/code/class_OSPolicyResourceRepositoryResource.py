from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceRepositoryResource(_messages.Message):
    """A resource that manages a package repository.

  Fields:
    apt: An Apt Repository.
    goo: A Goo Repository.
    yum: A Yum Repository.
    zypper: A Zypper Repository.
  """
    apt = _messages.MessageField('OSPolicyResourceRepositoryResourceAptRepository', 1)
    goo = _messages.MessageField('OSPolicyResourceRepositoryResourceGooRepository', 2)
    yum = _messages.MessageField('OSPolicyResourceRepositoryResourceYumRepository', 3)
    zypper = _messages.MessageField('OSPolicyResourceRepositoryResourceZypperRepository', 4)