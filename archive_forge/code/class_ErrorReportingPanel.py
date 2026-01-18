from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorReportingPanel(_messages.Message):
    """A widget that displays a list of error groups.

  Fields:
    projectNames: The resource name of the Google Cloud Platform project.
      Written as projects/{projectID} or projects/{projectNumber}, where
      {projectID} and {projectNumber} can be found in the Google Cloud console
      (https://support.google.com/cloud/answer/6158840).Examples: projects/my-
      project-123, projects/5551234.
    services: An identifier of the service, such as the name of the
      executable, job, or Google App Engine service name. This field is
      expected to have a low number of values that are relatively stable over
      time, as opposed to version, which can be changed whenever new code is
      deployed.Contains the service name for error reports extracted from
      Google App Engine logs or default if the App Engine default service is
      used.
    versions: Represents the source code version that the developer provided,
      which could represent a version label or a Git SHA-1 hash, for example.
      For App Engine standard environment, the version is set to the version
      of the app.
  """
    projectNames = _messages.StringField(1, repeated=True)
    services = _messages.StringField(2, repeated=True)
    versions = _messages.StringField(3, repeated=True)