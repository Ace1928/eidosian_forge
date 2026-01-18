from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ReservationsService(base_api.BaseApiService):
    """Service class for the reservations resource."""
    _NAME = 'reservations'

    def __init__(self, client):
        super(ComputeBeta.ReservationsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of reservations. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeReservationsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReservationAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.reservations.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/reservations', request_field='', request_type_name='ComputeReservationsAggregatedListRequest', response_type_name='ReservationAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified reservation.

      Args:
        request: (ComputeReservationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.reservations.delete', ordered_params=['project', 'zone', 'reservation'], path_params=['project', 'reservation', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/reservations/{reservation}', request_field='', request_type_name='ComputeReservationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves information about the specified reservation.

      Args:
        request: (ComputeReservationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Reservation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.reservations.get', ordered_params=['project', 'zone', 'reservation'], path_params=['project', 'reservation', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/reservations/{reservation}', request_field='', request_type_name='ComputeReservationsGetRequest', response_type_name='Reservation', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeReservationsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.reservations.getIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/zones/{zone}/reservations/{resource}/getIamPolicy', request_field='', request_type_name='ComputeReservationsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new reservation. For more information, read Reserving zonal resources.

      Args:
        request: (ComputeReservationsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.reservations.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/reservations', request_field='reservation', request_type_name='ComputeReservationsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """A list of all the reservations that have been configured for the specified project in specified zone.

      Args:
        request: (ComputeReservationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReservationList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.reservations.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/reservations', request_field='', request_type_name='ComputeReservationsListRequest', response_type_name='ReservationList', supports_download=False)

    def Resize(self, request, global_params=None):
        """Resizes the reservation (applicable to standalone reservations only). For more information, read Modifying reservations.

      Args:
        request: (ComputeReservationsResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.reservations.resize', ordered_params=['project', 'zone', 'reservation'], path_params=['project', 'reservation', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/reservations/{reservation}/resize', request_field='reservationsResizeRequest', request_type_name='ComputeReservationsResizeRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeReservationsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.reservations.setIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/reservations/{resource}/setIamPolicy', request_field='zoneSetPolicyRequest', request_type_name='ComputeReservationsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeReservationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.reservations.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/reservations/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeReservationsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Update share settings of the reservation.

      Args:
        request: (ComputeReservationsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.reservations.update', ordered_params=['project', 'zone', 'reservation'], path_params=['project', 'reservation', 'zone'], query_params=['paths', 'requestId', 'updateMask'], relative_path='projects/{project}/zones/{zone}/reservations/{reservation}', request_field='reservationResource', request_type_name='ComputeReservationsUpdateRequest', response_type_name='Operation', supports_download=False)