from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsCatalogTemplatesTagRequest(_messages.Message):
    """A DataflowProjectsCatalogTemplatesTagRequest object.

  Fields:
    modifyTemplateVersionTagRequest: A ModifyTemplateVersionTagRequest
      resource to be passed as the request body.
    name: Resource name includes project_id, display_name, and version_id.
      Updates by project_id(pid1), display_name(tid1), and version_id(vid1):
      Format: projects/{pid1}/catalogTemplates/{tid1@vid}
  """
    modifyTemplateVersionTagRequest = _messages.MessageField('ModifyTemplateVersionTagRequest', 1)
    name = _messages.StringField(2, required=True)