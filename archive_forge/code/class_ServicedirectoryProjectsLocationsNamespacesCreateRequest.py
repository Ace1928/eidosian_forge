from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesCreateRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesCreateRequest object.

  Fields:
    namespace: A Namespace resource to be passed as the request body.
    namespaceId: Required. The Resource ID must be 1-63 characters long, and
      comply with RFC1035. Specifically, the name must be 1-63 characters long
      and match the regular expression `[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?`
      which means the first character must be a lowercase letter, and all
      following characters must be a dash, lowercase letter, or digit, except
      the last character, which cannot be a dash.
    parent: Required. The resource name of the project and location the
      namespace will be created in.
  """
    namespace = _messages.MessageField('Namespace', 1)
    namespaceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)