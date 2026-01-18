from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
class ProxmoxKvmAnsible(ProxmoxAnsible):

    def get_vminfo(self, node, vmid, **kwargs):
        global results
        results = {}
        mac = {}
        devices = {}
        try:
            vm = self.proxmox_api.nodes(node).qemu(vmid).config.get()
        except Exception as e:
            self.module.fail_json(msg='Getting information for VM with vmid = %s failed with exception: %s' % (vmid, e))
        kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
        for k in list(kwargs.keys()):
            if isinstance(kwargs[k], dict):
                kwargs.update(kwargs[k])
                del kwargs[k]
        re_net = re.compile('net[0-9]')
        re_dev = re.compile('(virtio|ide|scsi|sata|efidisk)[0-9]')
        for k in kwargs.keys():
            if re_net.match(k):
                mac[k] = parse_mac(vm[k])
            elif re_dev.match(k):
                devices[k] = parse_dev(vm[k])
        results['mac'] = mac
        results['devices'] = devices
        results['vmid'] = int(vmid)

    def settings(self, vmid, node, **kwargs):
        proxmox_node = self.proxmox_api.nodes(node)
        kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
        return proxmox_node.qemu(vmid).config.set(**kwargs) is None

    def wait_for_task(self, node, taskid):
        timeout = self.module.params['timeout']
        if self.module.params['state'] == 'stopped':
            timeout += 10
        while timeout:
            if self.api_task_ok(node, taskid):
                time.sleep(1)
                return True
            timeout = timeout - 1
            if timeout == 0:
                break
            time.sleep(1)
        return False

    def create_vm(self, vmid, newid, node, name, memory, cpu, cores, sockets, update, update_unsafe, **kwargs):
        only_v4 = ['force', 'protection', 'skiplock']
        only_v6 = ['ciuser', 'cipassword', 'sshkeys', 'ipconfig', 'tags']
        valid_clone_params = ['format', 'full', 'pool', 'snapname', 'storage', 'target']
        clone_params = {}
        vm_args = '-serial unix:/var/run/qemu-server/{0}.serial,server,nowait'.format(vmid)
        proxmox_node = self.proxmox_api.nodes(node)
        kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
        kwargs.update(dict(([k, int(v)] for k, v in kwargs.items() if isinstance(v, bool))))
        version = self.version()
        pve_major_version = 3 if version < LooseVersion('4.0') else version.version[0]
        if pve_major_version < 4:
            for p in only_v4:
                if p in kwargs:
                    del kwargs[p]
        if pve_major_version < 6:
            for p in only_v6:
                if p in kwargs:
                    del kwargs[p]
        if 'sshkeys' in kwargs:
            urlencoded_ssh_keys = quote(kwargs['sshkeys'], safe='')
            kwargs['sshkeys'] = str(urlencoded_ssh_keys)
        if update:
            if update_unsafe is False:
                if 'virtio' in kwargs:
                    del kwargs['virtio']
                if 'sata' in kwargs:
                    del kwargs['sata']
                if 'scsi' in kwargs:
                    del kwargs['scsi']
                if 'ide' in kwargs:
                    del kwargs['ide']
                if 'efidisk0' in kwargs:
                    del kwargs['efidisk0']
                if 'tpmstate0' in kwargs:
                    del kwargs['tpmstate0']
                if 'net' in kwargs:
                    del kwargs['net']
            if 'force' in kwargs:
                del kwargs['force']
            if 'pool' in kwargs:
                del kwargs['pool']
        if 'efidisk0' in kwargs:
            if 'bios' not in kwargs or 'ovmf' != kwargs['bios']:
                self.module.fail_json(msg='efidisk0 cannot be used if bios is not set to ovmf. ')
        if 'efidisk0' in kwargs:
            efidisk0_str = ''
            hyphen_re = re.compile('_')
            if 'storage' in kwargs['efidisk0']:
                efidisk0_str += kwargs['efidisk0'].get('storage') + ':1,'
                kwargs['efidisk0'].pop('storage')
            efidisk0_str += ','.join([hyphen_re.sub('-', k) + '=' + str(v) for k, v in kwargs['efidisk0'].items() if 'storage' != k])
            kwargs['efidisk0'] = efidisk0_str
        if 'tpmstate0' in kwargs:
            kwargs['tpmstate0'] = '{storage}:1,version=v{version}'.format(storage=kwargs['tpmstate0'].get('storage'), version=kwargs['tpmstate0'].get('version'))
        for k in list(kwargs.keys()):
            if isinstance(kwargs[k], dict):
                kwargs.update(kwargs[k])
                del kwargs[k]
        if 'agent' in kwargs:
            try:
                kwargs['agent'] = int(boolean(kwargs['agent'], strict=True))
            except TypeError:
                pass
        if 'numa_enabled' in kwargs:
            kwargs['numa'] = kwargs['numa_enabled']
            del kwargs['numa_enabled']
        if 'nameservers' in self.module.params:
            nameservers = self.module.params.pop('nameservers')
            if nameservers:
                kwargs['nameserver'] = ' '.join(nameservers)
        if 'searchdomains' in self.module.params:
            searchdomains = self.module.params.pop('searchdomains')
            if searchdomains:
                kwargs['searchdomain'] = ' '.join(searchdomains)
        if 'tags' in kwargs:
            re_tag = re.compile('^[a-z0-9_][a-z0-9_\\-\\+\\.]*$')
            for tag in kwargs['tags']:
                if not re_tag.match(tag):
                    self.module.fail_json(msg='%s is not a valid tag' % tag)
            kwargs['tags'] = ','.join(kwargs['tags'])
        if self.module.params['api_user'] == 'root@pam' and self.module.params['args'] is None:
            if not update and self.module.params['proxmox_default_behavior'] == 'compatibility':
                kwargs['args'] = vm_args
        elif self.module.params['api_user'] == 'root@pam' and self.module.params['args'] is not None:
            kwargs['args'] = self.module.params['args']
        elif self.module.params['api_user'] != 'root@pam' and self.module.params['args'] is not None:
            self.module.fail_json(msg='args parameter require root@pam user. ')
        if self.module.params['api_user'] != 'root@pam' and self.module.params['skiplock'] is not None:
            self.module.fail_json(msg='skiplock parameter require root@pam user. ')
        if update:
            if proxmox_node.qemu(vmid).config.set(name=name, memory=memory, cpu=cpu, cores=cores, sockets=sockets, **kwargs) is None:
                return True
            else:
                return False
        elif self.module.params['clone'] is not None:
            for param in valid_clone_params:
                if self.module.params[param] is not None:
                    clone_params[param] = self.module.params[param]
            clone_params.update(dict(([k, int(v)] for k, v in clone_params.items() if isinstance(v, bool))))
            taskid = proxmox_node.qemu(vmid).clone.post(newid=newid, name=name, **clone_params)
        else:
            taskid = proxmox_node.qemu.create(vmid=vmid, name=name, memory=memory, cpu=cpu, cores=cores, sockets=sockets, **kwargs)
        if not self.wait_for_task(node, taskid):
            self.module.fail_json(msg='Reached timeout while waiting for creating VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
            return False
        return True

    def start_vm(self, vm):
        vmid = vm['vmid']
        proxmox_node = self.proxmox_api.nodes(vm['node'])
        taskid = proxmox_node.qemu(vmid).status.start.post()
        if not self.wait_for_task(vm['node'], taskid):
            self.module.fail_json(msg='Reached timeout while waiting for starting VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
            return False
        return True

    def stop_vm(self, vm, force, timeout):
        vmid = vm['vmid']
        proxmox_node = self.proxmox_api.nodes(vm['node'])
        taskid = proxmox_node.qemu(vmid).status.shutdown.post(forceStop=1 if force else 0, timeout=timeout)
        if not self.wait_for_task(vm['node'], taskid):
            self.module.fail_json(msg='Reached timeout while waiting for stopping VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
            return False
        return True

    def restart_vm(self, vm, force, **status):
        vmid = vm['vmid']
        try:
            proxmox_node = self.proxmox_api.nodes(vm['node'])
            taskid = proxmox_node.qemu(vmid).status.reset.post() if force else proxmox_node.qemu(vmid).status.reboot.post()
            if not self.wait_for_task(vm['node'], taskid):
                self.module.fail_json(msg='Reached timeout while waiting for rebooting VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
                return False
            return True
        except Exception as e:
            self.module.fail_json(vmid=vmid, msg='restarting of VM %s failed with exception: %s' % (vmid, e))
            return False

    def convert_to_template(self, vm, timeout, force):
        vmid = vm['vmid']
        try:
            proxmox_node = self.proxmox_api.nodes(vm['node'])
            if proxmox_node.qemu(vmid).status.current.get()['status'] == 'running' and force:
                self.stop_instance(vm, vmid, timeout, force)
            proxmox_node.qemu(vmid).template.post()
            return True
        except Exception as e:
            self.module.fail_json(vmid=vmid, msg='conversion of VM %s to template failed with exception: %s' % (vmid, e))
            return False

    def migrate_vm(self, vm, target_node):
        vmid = vm['vmid']
        proxmox_node = self.proxmox_api.nodes(vm['node'])
        taskid = proxmox_node.qemu(vmid).migrate.post(vmid=vmid, node=vm['node'], target=target_node, online=1)
        if not self.wait_for_task(vm['node'], taskid):
            self.module.fail_json(msg='Reached timeout while waiting for migrating VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
            return False
        return True