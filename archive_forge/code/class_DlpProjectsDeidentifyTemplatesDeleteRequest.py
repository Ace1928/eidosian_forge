from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsDeidentifyTemplatesDeleteRequest(_messages.Message):
    """A DlpProjectsDeidentifyTemplatesDeleteRequest object.

  Fields:
    name: Required. Resource name of the organization and deidentify template
      to be deleted, for example
      `organizations/433245324/deidentifyTemplates/432452342` or
      projects/project-id/deidentifyTemplates/432452342.
  """
    name = _messages.StringField(1, required=True)