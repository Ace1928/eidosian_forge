from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsTestMatricesCreateRequest(_messages.Message):
    """A TestingProjectsTestMatricesCreateRequest object.

  Fields:
    projectId: The GCE project under which this job will run.
    requestId: A string id used to detect duplicated requests. Ids are
      automatically scoped to a project, so users should ensure the ID is
      unique per-project. A UUID is recommended. Optional, but strongly
      recommended.
    testMatrix: A TestMatrix resource to be passed as the request body.
  """
    projectId = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    testMatrix = _messages.MessageField('TestMatrix', 3)