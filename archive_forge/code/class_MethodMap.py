from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MethodMap(_messages.Message):
    """Deployment Manager will call these methods during the events of
  creation/deletion/update/get/setIamPolicy

  Fields:
    create: The action identifier for the create method to be used for this
      collection
    delete: The action identifier for the delete method to be used for this
      collection
    get: The action identifier for the get method to be used for this
      collection
    setIamPolicy: The action identifier for the setIamPolicy method to be used
      for this collection
    update: The action identifier for the update method to be used for this
      collection
  """
    create = _messages.StringField(1)
    delete = _messages.StringField(2)
    get = _messages.StringField(3)
    setIamPolicy = _messages.StringField(4)
    update = _messages.StringField(5)