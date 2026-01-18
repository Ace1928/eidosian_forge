from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesListRequest(_messages.Message):
    """A MonitoringServicesListRequest object.

  Fields:
    filter: A filter specifying what Services to return. The filter supports
      filtering on a particular service-identifier type or one of its
      attributes.To filter on a particular service-identifier type, the
      identifier_case refers to which option in the identifier field is
      populated. For example, the filter identifier_case = "CUSTOM" would
      match all services with a value for the custom field. Valid options
      include "CUSTOM", "APP_ENGINE", "MESH_ISTIO", and the other options
      listed at https://cloud.google.com/monitoring/api/ref_v3/rest/v3/service
      s#ServiceTo filter on an attribute of a service-identifier type, apply
      the filter name by using the snake case of the service-identifier type
      and the attribute of that service-identifier type, and join the two with
      a period. For example, to filter by the meshUid field of the MeshIstio
      service-identifier type, you must filter on mesh_istio.mesh_uid = "123"
      to match all services with mesh UID "123". Service-identifier types and
      their attributes are described at
      https://cloud.google.com/monitoring/api/ref_v3/rest/v3/services#Service
    pageSize: A non-negative number that is the maximum number of results to
      return. When 0, use default page size.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
    parent: Required. Resource name of the parent containing the listed
      services, either a project
      (https://cloud.google.com/monitoring/api/v3#project_name) or a
      Monitoring Metrics Scope. The formats are:
      projects/[PROJECT_ID_OR_NUMBER] workspaces/[HOST_PROJECT_ID_OR_NUMBER]
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)