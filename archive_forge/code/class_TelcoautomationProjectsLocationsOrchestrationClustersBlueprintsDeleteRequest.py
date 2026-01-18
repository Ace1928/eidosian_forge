from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDeleteRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDeleteRequest
  object.

  Fields:
    name: Required. The name of blueprint to delete. Blueprint name should be
      in the format {blueprint_id}, if {blueprint_id}@{revision_id} is passed
      then the API throws invalid argument.
  """
    name = _messages.StringField(1, required=True)