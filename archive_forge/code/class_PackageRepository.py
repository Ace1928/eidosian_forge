from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PackageRepository(_messages.Message):
    """A package repository.

  Fields:
    apt: An Apt Repository.
    goo: A Goo Repository.
    yum: A Yum Repository.
    zypper: A Zypper Repository.
  """
    apt = _messages.MessageField('AptRepository', 1)
    goo = _messages.MessageField('GooRepository', 2)
    yum = _messages.MessageField('YumRepository', 3)
    zypper = _messages.MessageField('ZypperRepository', 4)