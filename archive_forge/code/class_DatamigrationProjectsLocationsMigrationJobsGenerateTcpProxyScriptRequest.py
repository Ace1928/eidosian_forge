from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsGenerateTcpProxyScriptRequest(_messages.Message):
    """A
  DatamigrationProjectsLocationsMigrationJobsGenerateTcpProxyScriptRequest
  object.

  Fields:
    generateTcpProxyScriptRequest: A GenerateTcpProxyScriptRequest resource to
      be passed as the request body.
    migrationJob: Name of the migration job resource to generate the TCP Proxy
      script.
  """
    generateTcpProxyScriptRequest = _messages.MessageField('GenerateTcpProxyScriptRequest', 1)
    migrationJob = _messages.StringField(2, required=True)