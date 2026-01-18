from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OwnerValue(_messages.Message):
    """The owner of the object. This will always be the uploader of the
    object.

    Fields:
      entity: The entity, in the form user-userId.
      entityId: The ID for the entity.
    """
    entity = _messages.StringField(1)
    entityId = _messages.StringField(2)