from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsCatalogTemplatesGetRequest(_messages.Message):
    """A DataflowProjectsCatalogTemplatesGetRequest object.

  Fields:
    name: Resource name includes project_id and display_name. version_id is
      optional. Get the latest TemplateVersion if version_id not set. Get by
      project_id(pid1) and display_name(tid1): Format:
      projects/{pid1}/catalogTemplates/{tid1} Get by project_id(pid1),
      display_name(tid1), and version_id(vid1): Format:
      projects/{pid1}/catalogTemplates/{tid1@vid}
  """
    name = _messages.StringField(1, required=True)