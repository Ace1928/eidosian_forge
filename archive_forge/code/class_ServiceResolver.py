from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceResolver(_messages.Message):
    """A ServiceResolver represents an EKM replica that can be reached within
  an EkmConnection.

  Fields:
    endpointFilter: Optional. The filter applied to the endpoints of the
      resolved service. If no filter is specified, all endpoints will be
      considered. An endpoint will be chosen arbitrarily from the filtered
      list for each request. For endpoint filter syntax and examples, see
      https://cloud.google.com/service-directory/docs/reference/rpc/google.clo
      ud.servicedirectory.v1#resolveservicerequest.
    hostname: Required. The hostname of the EKM replica used at TLS and HTTP
      layers.
    serverCertificates: Required. A list of leaf server certificates used to
      authenticate HTTPS connections to the EKM replica. Currently, a maximum
      of 10 Certificate is supported.
    serviceDirectoryService: Required. The resource name of the Service
      Directory service pointing to an EKM replica, in the format
      `projects/*/locations/*/namespaces/*/services/*`.
  """
    endpointFilter = _messages.StringField(1)
    hostname = _messages.StringField(2)
    serverCertificates = _messages.MessageField('Certificate', 3, repeated=True)
    serviceDirectoryService = _messages.StringField(4)