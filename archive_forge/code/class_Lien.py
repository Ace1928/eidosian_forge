from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Lien(_messages.Message):
    """A Lien represents an encumbrance on the actions that can be performed on
  a resource.

  Fields:
    createTime: The creation time of this Lien.
    name: A system-generated unique identifier for this Lien. Example:
      `liens/1234abcd`
    origin: A stable, user-visible/meaningful string identifying the origin of
      the Lien, intended to be inspected programmatically. Maximum length of
      200 characters. Example: 'compute.googleapis.com'
    parent: A reference to the resource this Lien is attached to. The server
      will validate the parent against those for which Liens are supported.
      Example: `projects/1234`
    reason: Concise user-visible strings indicating why an action cannot be
      performed on a resource. Maximum length of 200 characters. Example:
      'Holds production API key'
    restrictions: The types of operations which should be blocked as a result
      of this Lien. Each value should correspond to an IAM permission. The
      server will validate the permissions against those for which Liens are
      supported. An empty list is meaningless and will be rejected. Example:
      ['resourcemanager.projects.delete']
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    origin = _messages.StringField(3)
    parent = _messages.StringField(4)
    reason = _messages.StringField(5)
    restrictions = _messages.StringField(6, repeated=True)