from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
class VmwareDatastoreMaintenanceMgr(PyVmomi):

    def __init__(self, module):
        super(VmwareDatastoreMaintenanceMgr, self).__init__(module)
        datastore_name = self.params.get('datastore')
        cluster_name = self.params.get('cluster_name')
        datastore_cluster = self.params.get('datastore_cluster')
        self.datastore_objs = []
        if datastore_name:
            ds = self.find_datastore_by_name(datastore_name=datastore_name)
            if not ds:
                self.module.fail_json(msg='Failed to find datastore "%(datastore)s".' % self.params)
            self.datastore_objs = [ds]
        elif cluster_name:
            cluster = find_cluster_by_name(self.content, cluster_name)
            if not cluster:
                self.module.fail_json(msg='Failed to find cluster "%(cluster_name)s".' % self.params)
            self.datastore_objs = cluster.datastore
        elif datastore_cluster:
            datastore_cluster_obj = get_all_objs(self.content, [vim.StoragePod])
            if not datastore_cluster_obj:
                self.module.fail_json(msg='Failed to find datastore cluster "%(datastore_cluster)s".' % self.params)
            for datastore in datastore_cluster_obj.childEntity:
                self.datastore_objs.append(datastore)
        else:
            self.module.fail_json(msg="Please select one of 'cluster_name', 'datastore' or 'datastore_cluster'.")
        self.state = self.params.get('state')

    def ensure(self):
        datastore_results = dict()
        change_datastore_list = []
        for datastore in self.datastore_objs:
            changed = False
            if self.state == 'present' and datastore.summary.maintenanceMode != 'normal':
                datastore_results[datastore.name] = "Datastore '%s' is already in maintenance mode." % datastore.name
                break
            if self.state == 'absent' and datastore.summary.maintenanceMode == 'normal':
                datastore_results[datastore.name] = "Datastore '%s' is not in maintenance mode." % datastore.name
                break
            try:
                if self.state == 'present':
                    storage_replacement_result = datastore.DatastoreEnterMaintenanceMode()
                    task = storage_replacement_result.task
                else:
                    task = datastore.DatastoreExitMaintenanceMode_Task()
                success, result = wait_for_task(task)
                if success:
                    changed = True
                    if self.state == 'present':
                        datastore_results[datastore.name] = "Datastore '%s' entered in maintenance mode." % datastore.name
                    else:
                        datastore_results[datastore.name] = "Datastore '%s' exited from maintenance mode." % datastore.name
            except vim.fault.InvalidState as invalid_state:
                if self.state == 'present':
                    msg = "Unable to enter datastore '%s' in" % datastore.name
                else:
                    msg = "Unable to exit datastore '%s' from" % datastore.name
                msg += ' maintenance mode due to : %s' % to_native(invalid_state.msg)
                self.module.fail_json(msg=msg)
            except Exception as exc:
                if self.state == 'present':
                    msg = "Unable to enter datastore '%s' in" % datastore.name
                else:
                    msg = "Unable to exit datastore '%s' from" % datastore.name
                msg += ' maintenance mode due to generic exception : %s' % to_native(exc)
                self.module.fail_json(msg=msg)
            change_datastore_list.append(changed)
        changed = False
        if any(change_datastore_list):
            changed = True
        self.module.exit_json(changed=changed, datastore_status=datastore_results)