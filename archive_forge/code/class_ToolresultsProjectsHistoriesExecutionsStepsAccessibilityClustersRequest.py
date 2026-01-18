from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsAccessibilityClustersRequest(_messages.Message):
    """A
  ToolresultsProjectsHistoriesExecutionsStepsAccessibilityClustersRequest
  object.

  Fields:
    locale: The accepted format is the canonical Unicode format with hyphen as
      a delimiter. Language must be lowercase, Language Script - Capitalized,
      Region - UPPERCASE. See
      http://www.unicode.org/reports/tr35/#Unicode_locale_identifier for
      details. Required.
    name: A full resource name of the step. For example, projects/my-
      project/histories/bh.1234567890abcdef/executions/
      1234567890123456789/steps/bs.1234567890abcdef Required.
  """
    locale = _messages.StringField(1)
    name = _messages.StringField(2, required=True)