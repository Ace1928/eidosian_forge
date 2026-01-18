from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
class ProxmoxSnapAnsible(ProxmoxAnsible):

    def snapshot(self, vm, vmid):
        return getattr(self.proxmox_api.nodes(vm['node']), vm['type'])(vmid).snapshot

    def vmconfig(self, vm, vmid):
        return getattr(self.proxmox_api.nodes(vm['node']), vm['type'])(vmid).config

    def vmstatus(self, vm, vmid):
        return getattr(self.proxmox_api.nodes(vm['node']), vm['type'])(vmid).status

    def _container_mp_get(self, vm, vmid):
        cfg = self.vmconfig(vm, vmid).get()
        mountpoints = {}
        for key, value in cfg.items():
            if key.startswith('mp'):
                mountpoints[key] = value
        return mountpoints

    def _container_mp_disable(self, vm, vmid, timeout, unbind, mountpoints, vmstatus):
        if vmstatus == 'running':
            self.shutdown_instance(vm, vmid, timeout)
        self.vmconfig(vm, vmid).put(delete=' '.join(mountpoints))

    def _container_mp_restore(self, vm, vmid, timeout, unbind, mountpoints, vmstatus):
        self.vmconfig(vm, vmid).put(**mountpoints)
        if vmstatus == 'running':
            self.start_instance(vm, vmid, timeout)

    def start_instance(self, vm, vmid, timeout):
        taskid = self.vmstatus(vm, vmid).start.post()
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            timeout -= 1
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for VM to start. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False

    def shutdown_instance(self, vm, vmid, timeout):
        taskid = self.vmstatus(vm, vmid).shutdown.post()
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            timeout -= 1
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for VM to stop. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
        return False

    def snapshot_retention(self, vm, vmid, retention):
        snapshots = self.snapshot(vm, vmid).get()[:-1]
        if retention > 0 and len(snapshots) > retention:
            for snap in sorted(snapshots, key=lambda x: x['snaptime'])[:len(snapshots) - retention]:
                self.snapshot(vm, vmid)(snap['name']).delete()

    def snapshot_create(self, vm, vmid, timeout, snapname, description, vmstate, unbind, retention):
        if self.module.check_mode:
            return True
        if vm['type'] == 'lxc':
            if unbind is True:
                if self.module.params['api_user'] != 'root@pam' or not self.module.params['api_password']:
                    self.module.fail_json(msg='`unbind=True` requires authentication as `root@pam` with `api_password`, API tokens are not supported.')
                    return False
                mountpoints = self._container_mp_get(vm, vmid)
                vmstatus = self.vmstatus(vm, vmid).current().get()['status']
                if mountpoints:
                    self._container_mp_disable(vm, vmid, timeout, unbind, mountpoints, vmstatus)
            taskid = self.snapshot(vm, vmid).post(snapname=snapname, description=description)
        else:
            taskid = self.snapshot(vm, vmid).post(snapname=snapname, description=description, vmstate=int(vmstate))
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                break
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for creating VM snapshot. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
            timeout -= 1
        if vm['type'] == 'lxc' and unbind is True and mountpoints:
            self._container_mp_restore(vm, vmid, timeout, unbind, mountpoints, vmstatus)
        self.snapshot_retention(vm, vmid, retention)
        return timeout > 0

    def snapshot_remove(self, vm, vmid, timeout, snapname, force):
        if self.module.check_mode:
            return True
        taskid = self.snapshot(vm, vmid).delete(snapname, force=int(force))
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for removing VM snapshot. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
            timeout -= 1
        return False

    def snapshot_rollback(self, vm, vmid, timeout, snapname):
        if self.module.check_mode:
            return True
        taskid = self.snapshot(vm, vmid)(snapname).post('rollback')
        while timeout:
            if self.api_task_ok(vm['node'], taskid):
                return True
            if timeout == 0:
                self.module.fail_json(msg='Reached timeout while waiting for rolling back VM snapshot. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
            time.sleep(1)
            timeout -= 1
        return False