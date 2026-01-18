from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def FormatResponse(response, _):
    """Hook to modify gcloud describe output for maintenance windows."""
    modified_response = {}
    if response.authorizedNetwork:
        modified_response['authorizedNetwork'] = response.authorizedNetwork
    if response.createTime:
        modified_response['createTime'] = response.createTime
    if response.discoveryEndpoint:
        modified_response['discoveryEndpoint'] = response.discoveryEndpoint
    if response.displayName:
        modified_response['displayName'] = response.displayName
    if response.maintenanceSchedule:
        modified_response['maintenanceSchedule'] = response.maintenanceSchedule
    if response.memcacheFullVersion:
        modified_response['memcacheFullVersion'] = response.memcacheFullVersion
    if response.memcacheNodes:
        modified_response['memcacheNodes'] = response.memcacheNodes
    if response.memcacheVersion:
        modified_response['memcacheVersion'] = response.memcacheVersion
    if response.name:
        modified_response['name'] = response.name
    if response.nodeConfig:
        modified_response['nodeConfig'] = response.nodeConfig
    if response.nodeCount:
        modified_response['nodeCount'] = response.nodeCount
    if response.parameters:
        modified_response['parameters'] = response.parameters
    if response.state:
        modified_response['state'] = response.state
    if response.updateTime:
        modified_response['updateTime'] = response.updateTime
    if response.zones:
        modified_response['zones'] = response.zones
    if response.maintenancePolicy:
        modified_mw_policy = {}
        modified_mw_policy['createTime'] = response.maintenancePolicy.createTime
        modified_mw_policy['updateTime'] = response.maintenancePolicy.updateTime
        mwlist = response.maintenancePolicy.weeklyMaintenanceWindow
        modified_mwlist = []
        for mw in mwlist:
            item = {}
            duration_secs = int(mw.duration[:-1])
            duration_mins = int(duration_secs / 60)
            item['day'] = mw.day
            item['hour'] = mw.startTime.hours
            item['duration'] = six.text_type(duration_mins) + ' minutes'
            modified_mwlist.append(item)
        modified_mw_policy['maintenanceWindow'] = modified_mwlist
        modified_response['maintenancePolicy'] = modified_mw_policy
    return modified_response