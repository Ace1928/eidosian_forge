from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsCatalogTemplatesTemplateVersionsCreateRequest(_messages.Message):
    """A DataflowProjectsCatalogTemplatesTemplateVersionsCreateRequest object.

  Fields:
    createTemplateVersionRequest: A CreateTemplateVersionRequest resource to
      be passed as the request body.
    parent: The parent project and template that the TemplateVersion will be
      created under. Create using project_id(pid1) and display_name(tid1).
      Format: projects/{pid1}/catalogTemplates/{tid1}
  """
    createTemplateVersionRequest = _messages.MessageField('CreateTemplateVersionRequest', 1)
    parent = _messages.StringField(2, required=True)