from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionInstanceGroupManagersService(base_api.BaseApiService):
    """Service class for the regionInstanceGroupManagers resource."""
    _NAME = 'regionInstanceGroupManagers'

    def __init__(self, client):
        super(ComputeBeta.RegionInstanceGroupManagersService, self).__init__(client)
        self._upload_configs = {}

    def AbandonInstances(self, request, global_params=None):
        """Flags the specified instances to be immediately removed from the managed instance group. Abandoning an instance does not delete the instance, but it does remove the instance from any target pools that are applied by the managed instance group. This method reduces the targetSize of the managed instance group by the number of instances that you abandon. This operation is marked as DONE when the action is scheduled even if the instances have not yet been removed from the group. You must separately verify the status of the abandoning action with the listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersAbandonInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AbandonInstances')
        return self._RunMethod(config, request, global_params=global_params)
    AbandonInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.abandonInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/abandonInstances', request_field='regionInstanceGroupManagersAbandonInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersAbandonInstancesRequest', response_type_name='Operation', supports_download=False)

    def ApplyUpdatesToInstances(self, request, global_params=None):
        """Apply updates to selected instances the managed instance group.

      Args:
        request: (ComputeRegionInstanceGroupManagersApplyUpdatesToInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ApplyUpdatesToInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ApplyUpdatesToInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.applyUpdatesToInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/applyUpdatesToInstances', request_field='regionInstanceGroupManagersApplyUpdatesRequest', request_type_name='ComputeRegionInstanceGroupManagersApplyUpdatesToInstancesRequest', response_type_name='Operation', supports_download=False)

    def CreateInstances(self, request, global_params=None):
        """Creates instances with per-instance configurations in this regional managed instance group. Instances are created using the current instance template. The create instances operation is marked DONE if the createInstances request is successful. The underlying actions take additional time. You must separately verify the status of the creating or actions with the listmanagedinstances method.

      Args:
        request: (ComputeRegionInstanceGroupManagersCreateInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateInstances')
        return self._RunMethod(config, request, global_params=global_params)
    CreateInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.createInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/createInstances', request_field='regionInstanceGroupManagersCreateInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersCreateInstancesRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified managed instance group and all of the instances in that group.

      Args:
        request: (ComputeRegionInstanceGroupManagersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionInstanceGroupManagers.delete', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}', request_field='', request_type_name='ComputeRegionInstanceGroupManagersDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be immediately deleted. The instances are also removed from any target pools of which they were a member. This method reduces the targetSize of the managed instance group by the number of instances that you delete. The deleteInstances operation is marked DONE if the deleteInstances request is successful. The underlying actions take additional time. You must separately verify the status of the deleting action with the listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersDeleteInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteInstances')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.deleteInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/deleteInstances', request_field='regionInstanceGroupManagersDeleteInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersDeleteInstancesRequest', response_type_name='Operation', supports_download=False)

    def DeletePerInstanceConfigs(self, request, global_params=None):
        """Deletes selected per-instance configurations for the managed instance group.

      Args:
        request: (ComputeRegionInstanceGroupManagersDeletePerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeletePerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    DeletePerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.deletePerInstanceConfigs', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/deletePerInstanceConfigs', request_field='regionInstanceGroupManagerDeleteInstanceConfigReq', request_type_name='ComputeRegionInstanceGroupManagersDeletePerInstanceConfigsRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns all of the details about the specified managed instance group.

      Args:
        request: (ComputeRegionInstanceGroupManagersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManager) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceGroupManagers.get', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}', request_field='', request_type_name='ComputeRegionInstanceGroupManagersGetRequest', response_type_name='InstanceGroupManager', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a managed instance group using the information that you specify in the request. After the group is created, instances in the group are created using the specified instance template. This operation is marked as DONE when the group is created even if the instances in the group have not yet been created. You must separately verify the status of the individual instances with the listmanagedinstances method. A regional managed instance group can contain up to 2000 instances.

      Args:
        request: (ComputeRegionInstanceGroupManagersInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers', request_field='instanceGroupManager', request_type_name='ComputeRegionInstanceGroupManagersInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of managed instance groups that are contained within the specified region.

      Args:
        request: (ComputeRegionInstanceGroupManagersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagerList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceGroupManagers.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers', request_field='', request_type_name='ComputeRegionInstanceGroupManagersListRequest', response_type_name='RegionInstanceGroupManagerList', supports_download=False)

    def ListErrors(self, request, global_params=None):
        """Lists all errors thrown by actions on instances for a given regional managed instance group. The filter and orderBy query parameters are not supported.

      Args:
        request: (ComputeRegionInstanceGroupManagersListErrorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagersListErrorsResponse) The response message.
      """
        config = self.GetMethodConfig('ListErrors')
        return self._RunMethod(config, request, global_params=global_params)
    ListErrors.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceGroupManagers.listErrors', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/listErrors', request_field='', request_type_name='ComputeRegionInstanceGroupManagersListErrorsRequest', response_type_name='RegionInstanceGroupManagersListErrorsResponse', supports_download=False)

    def ListManagedInstances(self, request, global_params=None):
        """Lists the instances in the managed instance group and instances that are scheduled to be created. The list includes any current actions that the group has scheduled for its instances. The orderBy query parameter is not supported. The `pageToken` query parameter is supported only if the group's `listManagedInstancesResults` field is set to `PAGINATED`.

      Args:
        request: (ComputeRegionInstanceGroupManagersListManagedInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagersListInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('ListManagedInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ListManagedInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.listManagedInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/listManagedInstances', request_field='', request_type_name='ComputeRegionInstanceGroupManagersListManagedInstancesRequest', response_type_name='RegionInstanceGroupManagersListInstancesResponse', supports_download=False)

    def ListPerInstanceConfigs(self, request, global_params=None):
        """Lists all of the per-instance configurations defined for the managed instance group. The orderBy query parameter is not supported.

      Args:
        request: (ComputeRegionInstanceGroupManagersListPerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagersListInstanceConfigsResp) The response message.
      """
        config = self.GetMethodConfig('ListPerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    ListPerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.listPerInstanceConfigs', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/listPerInstanceConfigs', request_field='', request_type_name='ComputeRegionInstanceGroupManagersListPerInstanceConfigsRequest', response_type_name='RegionInstanceGroupManagersListInstanceConfigsResp', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a managed instance group using the information that you specify in the request. This operation is marked as DONE when the group is patched even if the instances in the group are still in the process of being patched. You must separately verify the status of the individual instances with the listmanagedinstances method. This method supports PATCH semantics and uses the JSON merge patch format and processing rules. If you update your group to specify a new template or instance configuration, it's possible that your intended specification for each VM in the group is different from the current state of that VM. To learn how to apply an updated configuration to the VMs in a MIG, see Updating instances in a MIG.

      Args:
        request: (ComputeRegionInstanceGroupManagersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionInstanceGroupManagers.patch', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}', request_field='instanceGroupManagerResource', request_type_name='ComputeRegionInstanceGroupManagersPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchPerInstanceConfigs(self, request, global_params=None):
        """Inserts or patches per-instance configurations for the managed instance group. perInstanceConfig.name serves as a key used to distinguish whether to perform insert or patch.

      Args:
        request: (ComputeRegionInstanceGroupManagersPatchPerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchPerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    PatchPerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.patchPerInstanceConfigs', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/patchPerInstanceConfigs', request_field='regionInstanceGroupManagerPatchInstanceConfigReq', request_type_name='ComputeRegionInstanceGroupManagersPatchPerInstanceConfigsRequest', response_type_name='Operation', supports_download=False)

    def RecreateInstances(self, request, global_params=None):
        """Flags the specified VM instances in the managed instance group to be immediately recreated. Each instance is recreated using the group's current configuration. This operation is marked as DONE when the flag is set even if the instances have not yet been recreated. You must separately verify the status of each instance by checking its currentAction field; for more information, see Checking the status of managed instances. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersRecreateInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RecreateInstances')
        return self._RunMethod(config, request, global_params=global_params)
    RecreateInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.recreateInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/recreateInstances', request_field='regionInstanceGroupManagersRecreateRequest', request_type_name='ComputeRegionInstanceGroupManagersRecreateInstancesRequest', response_type_name='Operation', supports_download=False)

    def Resize(self, request, global_params=None):
        """Changes the intended size of the managed instance group. If you increase the size, the group creates new instances using the current instance template. If you decrease the size, the group deletes one or more instances. The resize operation is marked DONE if the resize request is successful. The underlying actions take additional time. You must separately verify the status of the creating or deleting actions with the listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted.

      Args:
        request: (ComputeRegionInstanceGroupManagersResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.resize', ordered_params=['project', 'region', 'instanceGroupManager', 'size'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId', 'size'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/resize', request_field='', request_type_name='ComputeRegionInstanceGroupManagersResizeRequest', response_type_name='Operation', supports_download=False)

    def ResizeAdvanced(self, request, global_params=None):
        """Resizes the regional managed instance group with advanced configuration options like disabling creation retries. This is an extended version of the resize method. If you increase the size, the group creates new instances using the current instance template. If you decrease the size, the group deletes one or more instances. The resize operation is marked DONE if the resize request is successful. The underlying actions take additional time. You must separately verify the status of the creating or deleting actions with the get or listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted.

      Args:
        request: (ComputeRegionInstanceGroupManagersResizeAdvancedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResizeAdvanced')
        return self._RunMethod(config, request, global_params=global_params)
    ResizeAdvanced.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.resizeAdvanced', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/resizeAdvanced', request_field='regionInstanceGroupManagersResizeAdvancedRequest', request_type_name='ComputeRegionInstanceGroupManagersResizeAdvancedRequest', response_type_name='Operation', supports_download=False)

    def ResumeInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be resumed. This method increases the targetSize and decreases the targetSuspendedSize of the managed instance group by the number of instances that you resume. The resumeInstances operation is marked DONE if the resumeInstances request is successful. The underlying actions take additional time. You must separately verify the status of the RESUMING action with the listmanagedinstances method. In this request, you can only specify instances that are suspended. For example, if an instance was previously suspended using the suspendInstances method, it can be resumed using the resumeInstances method. If a health check is attached to the managed instance group, the specified instances will be verified as healthy after they are resumed. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersResumeInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResumeInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ResumeInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.resumeInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/resumeInstances', request_field='regionInstanceGroupManagersResumeInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersResumeInstancesRequest', response_type_name='Operation', supports_download=False)

    def SetAutoHealingPolicies(self, request, global_params=None):
        """Modifies the autohealing policy for the instances in this managed instance group. [Deprecated] This method is deprecated. Use regionInstanceGroupManagers.patch instead.

      Args:
        request: (ComputeRegionInstanceGroupManagersSetAutoHealingPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetAutoHealingPolicies')
        return self._RunMethod(config, request, global_params=global_params)
    SetAutoHealingPolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.setAutoHealingPolicies', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/setAutoHealingPolicies', request_field='regionInstanceGroupManagersSetAutoHealingRequest', request_type_name='ComputeRegionInstanceGroupManagersSetAutoHealingPoliciesRequest', response_type_name='Operation', supports_download=False)

    def SetInstanceTemplate(self, request, global_params=None):
        """Sets the instance template to use when creating new instances or recreating instances in this group. Existing instances are not affected.

      Args:
        request: (ComputeRegionInstanceGroupManagersSetInstanceTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetInstanceTemplate')
        return self._RunMethod(config, request, global_params=global_params)
    SetInstanceTemplate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.setInstanceTemplate', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/setInstanceTemplate', request_field='regionInstanceGroupManagersSetTemplateRequest', request_type_name='ComputeRegionInstanceGroupManagersSetInstanceTemplateRequest', response_type_name='Operation', supports_download=False)

    def SetTargetPools(self, request, global_params=None):
        """Modifies the target pools to which all new instances in this group are assigned. Existing instances in the group are not affected.

      Args:
        request: (ComputeRegionInstanceGroupManagersSetTargetPoolsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetTargetPools')
        return self._RunMethod(config, request, global_params=global_params)
    SetTargetPools.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.setTargetPools', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/setTargetPools', request_field='regionInstanceGroupManagersSetTargetPoolsRequest', request_type_name='ComputeRegionInstanceGroupManagersSetTargetPoolsRequest', response_type_name='Operation', supports_download=False)

    def StartInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be started. This method increases the targetSize and decreases the targetStoppedSize of the managed instance group by the number of instances that you start. The startInstances operation is marked DONE if the startInstances request is successful. The underlying actions take additional time. You must separately verify the status of the STARTING action with the listmanagedinstances method. In this request, you can only specify instances that are stopped. For example, if an instance was previously stopped using the stopInstances method, it can be started using the startInstances method. If a health check is attached to the managed instance group, the specified instances will be verified as healthy after they are started. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersStartInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartInstances')
        return self._RunMethod(config, request, global_params=global_params)
    StartInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.startInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/startInstances', request_field='regionInstanceGroupManagersStartInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersStartInstancesRequest', response_type_name='Operation', supports_download=False)

    def StopInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be immediately stopped. You can only specify instances that are running in this request. This method reduces the targetSize and increases the targetStoppedSize of the managed instance group by the number of instances that you stop. The stopInstances operation is marked DONE if the stopInstances request is successful. The underlying actions take additional time. You must separately verify the status of the STOPPING action with the listmanagedinstances method. If the standbyPolicy.initialDelaySec field is set, the group delays stopping the instances until initialDelaySec have passed from instance.creationTimestamp (that is, when the instance was created). This delay gives your application time to set itself up and initialize on the instance. If more than initialDelaySec seconds have passed since instance.creationTimestamp when this method is called, there will be zero delay. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is stopped. Stopped instances can be started using the startInstances method. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersStopInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopInstances')
        return self._RunMethod(config, request, global_params=global_params)
    StopInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.stopInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/stopInstances', request_field='regionInstanceGroupManagersStopInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersStopInstancesRequest', response_type_name='Operation', supports_download=False)

    def SuspendInstances(self, request, global_params=None):
        """Flags the specified instances in the managed instance group to be immediately suspended. You can only specify instances that are running in this request. This method reduces the targetSize and increases the targetSuspendedSize of the managed instance group by the number of instances that you suspend. The suspendInstances operation is marked DONE if the suspendInstances request is successful. The underlying actions take additional time. You must separately verify the status of the SUSPENDING action with the listmanagedinstances method. If the standbyPolicy.initialDelaySec field is set, the group delays suspension of the instances until initialDelaySec have passed from instance.creationTimestamp (that is, when the instance was created). This delay gives your application time to set itself up and initialize on the instance. If more than initialDelaySec seconds have passed since instance.creationTimestamp when this method is called, there will be zero delay. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is suspended. Suspended instances can be resumed using the resumeInstances method. You can specify a maximum of 1000 instances with this method per request.

      Args:
        request: (ComputeRegionInstanceGroupManagersSuspendInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SuspendInstances')
        return self._RunMethod(config, request, global_params=global_params)
    SuspendInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.suspendInstances', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/suspendInstances', request_field='regionInstanceGroupManagersSuspendInstancesRequest', request_type_name='ComputeRegionInstanceGroupManagersSuspendInstancesRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionInstanceGroupManagersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionInstanceGroupManagersTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a managed instance group using the information that you specify in the request. This operation is marked as DONE when the group is updated even if the instances in the group have not yet been updated. You must separately verify the status of the individual instances with the listmanagedinstances method. If you update your group to specify a new template or instance configuration, it's possible that your intended specification for each VM in the group is different from the current state of that VM. To learn how to apply an updated configuration to the VMs in a MIG, see Updating instances in a MIG.

      Args:
        request: (ComputeRegionInstanceGroupManagersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.regionInstanceGroupManagers.update', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}', request_field='instanceGroupManagerResource', request_type_name='ComputeRegionInstanceGroupManagersUpdateRequest', response_type_name='Operation', supports_download=False)

    def UpdatePerInstanceConfigs(self, request, global_params=None):
        """Inserts or updates per-instance configurations for the managed instance group. perInstanceConfig.name serves as a key used to distinguish whether to perform insert or patch.

      Args:
        request: (ComputeRegionInstanceGroupManagersUpdatePerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdatePerInstanceConfigs')
        return self._RunMethod(config, request, global_params=global_params)
    UpdatePerInstanceConfigs.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroupManagers.updatePerInstanceConfigs', ordered_params=['project', 'region', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroupManagers/{instanceGroupManager}/updatePerInstanceConfigs', request_field='regionInstanceGroupManagerUpdateInstanceConfigReq', request_type_name='ComputeRegionInstanceGroupManagersUpdatePerInstanceConfigsRequest', response_type_name='Operation', supports_download=False)