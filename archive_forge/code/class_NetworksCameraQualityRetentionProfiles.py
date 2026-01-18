from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksCameraQualityRetentionProfiles(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), motionBasedRetentionEnabled=params.get('motionBasedRetentionEnabled'), restrictedBandwidthModeEnabled=params.get('restrictedBandwidthModeEnabled'), audioRecordingEnabled=params.get('audioRecordingEnabled'), cloudArchiveEnabled=params.get('cloudArchiveEnabled'), motionDetectorVersion=params.get('motionDetectorVersion'), scheduleId=params.get('scheduleId'), maxRetentionDays=params.get('maxRetentionDays'), videoSettings=params.get('videoSettings'), networkId=params.get('networkId'), qualityRetentionProfileId=params.get('qualityRetentionProfileId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('qualityRetentionProfileId') is not None or self.new_object.get('quality_retention_profile_id') is not None:
            new_object_params['qualityRetentionProfileId'] = self.new_object.get('qualityRetentionProfileId') or self.new_object.get('quality_retention_profile_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('motionBasedRetentionEnabled') is not None or self.new_object.get('motion_based_retention_enabled') is not None:
            new_object_params['motionBasedRetentionEnabled'] = self.new_object.get('motionBasedRetentionEnabled')
        if self.new_object.get('restrictedBandwidthModeEnabled') is not None or self.new_object.get('restricted_bandwidth_mode_enabled') is not None:
            new_object_params['restrictedBandwidthModeEnabled'] = self.new_object.get('restrictedBandwidthModeEnabled')
        if self.new_object.get('audioRecordingEnabled') is not None or self.new_object.get('audio_recording_enabled') is not None:
            new_object_params['audioRecordingEnabled'] = self.new_object.get('audioRecordingEnabled')
        if self.new_object.get('cloudArchiveEnabled') is not None or self.new_object.get('cloud_archive_enabled') is not None:
            new_object_params['cloudArchiveEnabled'] = self.new_object.get('cloudArchiveEnabled')
        if self.new_object.get('motionDetectorVersion') is not None or self.new_object.get('motion_detector_version') is not None:
            new_object_params['motionDetectorVersion'] = self.new_object.get('motionDetectorVersion') or self.new_object.get('motion_detector_version')
        if self.new_object.get('scheduleId') is not None or self.new_object.get('schedule_id') is not None:
            new_object_params['scheduleId'] = self.new_object.get('scheduleId') or self.new_object.get('schedule_id')
        if self.new_object.get('maxRetentionDays') is not None or self.new_object.get('max_retention_days') is not None:
            new_object_params['maxRetentionDays'] = self.new_object.get('maxRetentionDays') or self.new_object.get('max_retention_days')
        if self.new_object.get('videoSettings') is not None or self.new_object.get('video_settings') is not None:
            new_object_params['videoSettings'] = self.new_object.get('videoSettings') or self.new_object.get('video_settings')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('qualityRetentionProfileId') is not None or self.new_object.get('quality_retention_profile_id') is not None:
            new_object_params['qualityRetentionProfileId'] = self.new_object.get('qualityRetentionProfileId') or self.new_object.get('quality_retention_profile_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('motionBasedRetentionEnabled') is not None or self.new_object.get('motion_based_retention_enabled') is not None:
            new_object_params['motionBasedRetentionEnabled'] = self.new_object.get('motionBasedRetentionEnabled')
        if self.new_object.get('restrictedBandwidthModeEnabled') is not None or self.new_object.get('restricted_bandwidth_mode_enabled') is not None:
            new_object_params['restrictedBandwidthModeEnabled'] = self.new_object.get('restrictedBandwidthModeEnabled')
        if self.new_object.get('audioRecordingEnabled') is not None or self.new_object.get('audio_recording_enabled') is not None:
            new_object_params['audioRecordingEnabled'] = self.new_object.get('audioRecordingEnabled')
        if self.new_object.get('cloudArchiveEnabled') is not None or self.new_object.get('cloud_archive_enabled') is not None:
            new_object_params['cloudArchiveEnabled'] = self.new_object.get('cloudArchiveEnabled')
        if self.new_object.get('motionDetectorVersion') is not None or self.new_object.get('motion_detector_version') is not None:
            new_object_params['motionDetectorVersion'] = self.new_object.get('motionDetectorVersion') or self.new_object.get('motion_detector_version')
        if self.new_object.get('scheduleId') is not None or self.new_object.get('schedule_id') is not None:
            new_object_params['scheduleId'] = self.new_object.get('scheduleId') or self.new_object.get('schedule_id')
        if self.new_object.get('maxRetentionDays') is not None or self.new_object.get('max_retention_days') is not None:
            new_object_params['maxRetentionDays'] = self.new_object.get('maxRetentionDays') or self.new_object.get('max_retention_days')
        if self.new_object.get('videoSettings') is not None or self.new_object.get('video_settings') is not None:
            new_object_params['videoSettings'] = self.new_object.get('videoSettings') or self.new_object.get('video_settings')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('qualityRetentionProfileId') is not None or self.new_object.get('quality_retention_profile_id') is not None:
            new_object_params['qualityRetentionProfileId'] = self.new_object.get('qualityRetentionProfileId') or self.new_object.get('quality_retention_profile_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='camera', function='getNetworkCameraQualityRetentionProfiles', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='camera', function='getNetworkCameraQualityRetentionProfile', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'qualityRetentionProfileId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('quality_retention_profile_id') or self.new_object.get('qualityRetentionProfileId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('qualityRetentionProfileId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(qualityRetentionProfileId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('motionBasedRetentionEnabled', 'motionBasedRetentionEnabled'), ('restrictedBandwidthModeEnabled', 'restrictedBandwidthModeEnabled'), ('audioRecordingEnabled', 'audioRecordingEnabled'), ('cloudArchiveEnabled', 'cloudArchiveEnabled'), ('motionDetectorVersion', 'motionDetectorVersion'), ('scheduleId', 'scheduleId'), ('maxRetentionDays', 'maxRetentionDays'), ('videoSettings', 'videoSettings'), ('networkId', 'networkId'), ('qualityRetentionProfileId', 'qualityRetentionProfileId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='camera', function='createNetworkCameraQualityRetentionProfile', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('qualityRetentionProfileId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('qualityRetentionProfileId')
            if id_:
                self.new_object.update(dict(qualityRetentionProfileId=id_))
        result = self.meraki.exec_meraki(family='camera', function='updateNetworkCameraQualityRetentionProfile', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('qualityRetentionProfileId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('qualityRetentionProfileId')
            if id_:
                self.new_object.update(dict(qualityRetentionProfileId=id_))
        result = self.meraki.exec_meraki(family='camera', function='deleteNetworkCameraQualityRetentionProfile', params=self.delete_by_id_params())
        return result