from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareDatastore(PyVmomi):

    def __init__(self, module):
        super(VMwareDatastore, self).__init__(module)
        if self.module.params.get('congestion_threshold_percentage') not in range(50, 101):
            self.module.fail_json(msg='Congestion Threshold should be between 50% and 100%.')
        self.datacenter_name = self.module.params.get('datacenter')
        if self.datacenter_name:
            self.datacenter = self.find_datacenter_by_name(self.datacenter_name)
            if self.datacenter is None:
                self.module.fail_json(msg='Datacenter %s does not exist.' % self.datacenter_name)
        else:
            self.datacenter = None
        self.datastore_name = self.module.params.get('name')
        self.datastore = self.find_datastore_by_name(self.datastore_name, self.datacenter)
        if self.datastore is None:
            self.module.fail_json(msg='Datastore %s does not exist.' % self.name)
        self.storageResourceManager = self.content.storageResourceManager

    def check_config_diff(self):
        """
        Check configuration diff
        Returns: True if there is diff, else False
        """
        iormConfiguration = self.datastore.iormConfiguration
        conf_statsAggregationDisabled = not self.module.params.get('statistic_collection')
        if self.module.params.get('storage_io_control') == 'enable_io_statistics':
            if self.module.params.get('congestion_threshold_manual') is not None:
                conf_congestionThresholdMode = 'manual'
                conf_congestionThreshold = self.module.params.get('congestion_threshold_manual')
                conf_percentOfPeakThroughput = iormConfiguration.percentOfPeakThroughput
            else:
                conf_congestionThresholdMode = 'automatic'
                conf_percentOfPeakThroughput = self.module.params.get('congestion_threshold_percentage')
                conf_congestionThreshold = iormConfiguration.congestionThreshold
            if iormConfiguration.enabled and iormConfiguration.statsCollectionEnabled and (iormConfiguration.statsAggregationDisabled == conf_statsAggregationDisabled) and (iormConfiguration.congestionThresholdMode == conf_congestionThresholdMode) and (iormConfiguration.congestionThreshold == conf_congestionThreshold) and (iormConfiguration.percentOfPeakThroughput == conf_percentOfPeakThroughput):
                return False
            else:
                return True
        elif self.module.params.get('storage_io_control') == 'enable_statistics':
            if not iormConfiguration.enabled and iormConfiguration.statsCollectionEnabled and (iormConfiguration.statsAggregationDisabled == conf_statsAggregationDisabled):
                return False
            else:
                return True
        elif self.module.params.get('storage_io_control') == 'disable':
            if not iormConfiguration.enabled and (not iormConfiguration.statsCollectionEnabled):
                return False
            else:
                return True

    def configure(self):
        """
        Manage configuration
        """
        changed = self.check_config_diff()
        if changed:
            if not self.module.check_mode:
                config_spec = vim.StorageResourceManager.IORMConfigSpec()
                iormConfiguration = self.datastore.iormConfiguration
                conf_statsAggregationDisabled = not self.module.params.get('statistic_collection')
                if self.module.params.get('storage_io_control') == 'enable_io_statistics':
                    if self.module.params.get('congestion_threshold_manual') is not None:
                        config_spec.congestionThresholdMode = 'manual'
                        config_spec.congestionThreshold = self.module.params.get('congestion_threshold_manual')
                        config_spec.percentOfPeakThroughput = iormConfiguration.percentOfPeakThroughput
                    else:
                        config_spec.congestionThresholdMode = 'automatic'
                        config_spec.percentOfPeakThroughput = self.module.params.get('congestion_threshold_percentage')
                        config_spec.congestionThreshold = iormConfiguration.congestionThreshold
                    config_spec.enabled = True
                    config_spec.statsCollectionEnabled = True
                    config_spec.statsAggregationDisabled = conf_statsAggregationDisabled
                elif self.module.params.get('storage_io_control') == 'enable_statistics':
                    config_spec.enabled = False
                    config_spec.statsCollectionEnabled = True
                    config_spec.statsAggregationDisabled = conf_statsAggregationDisabled
                elif self.module.params.get('storage_io_control') == 'disable':
                    config_spec.enabled = False
                    config_spec.statsCollectionEnabled = False
                try:
                    task = self.storageResourceManager.ConfigureDatastoreIORM_Task(self.datastore, config_spec)
                    changed, result = wait_for_task(task)
                except TaskError as generic_exc:
                    self.module.fail_json(msg=to_native(generic_exc))
                except Exception as task_e:
                    self.module.fail_json(msg=to_native(task_e))
            else:
                changed = True
        results = dict(changed=changed)
        self.module.exit_json(**results)