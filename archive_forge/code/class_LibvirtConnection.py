from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
class LibvirtConnection(object):

    def __init__(self, uri, module):
        self.module = module
        cmd = 'uname -r'
        rc, stdout, stderr = self.module.run_command(cmd)
        if 'xen' in stdout:
            conn = libvirt.open(None)
        elif 'esx' in uri:
            auth = [[libvirt.VIR_CRED_AUTHNAME, libvirt.VIR_CRED_NOECHOPROMPT], [], None]
            conn = libvirt.openAuth(uri, auth)
        else:
            conn = libvirt.open(uri)
        if not conn:
            raise Exception('hypervisor connection failure')
        self.conn = conn

    def find_vm(self, vmid):
        """
        Extra bonus feature: vmid = -1 returns a list of everything
        """
        vms = self.conn.listAllDomains()
        if vmid == -1:
            return vms
        for vm in vms:
            if vm.name() == vmid:
                return vm
        raise VMNotFound('virtual machine %s not found' % vmid)

    def shutdown(self, vmid):
        return self.find_vm(vmid).shutdown()

    def pause(self, vmid):
        return self.suspend(vmid)

    def unpause(self, vmid):
        return self.resume(vmid)

    def suspend(self, vmid):
        return self.find_vm(vmid).suspend()

    def resume(self, vmid):
        return self.find_vm(vmid).resume()

    def create(self, vmid):
        return self.find_vm(vmid).create()

    def destroy(self, vmid):
        return self.find_vm(vmid).destroy()

    def undefine(self, vmid, flag):
        return self.find_vm(vmid).undefineFlags(flag)

    def get_status2(self, vm):
        state = vm.info()[0]
        return VIRT_STATE_NAME_MAP.get(state, 'unknown')

    def get_status(self, vmid):
        state = self.find_vm(vmid).info()[0]
        return VIRT_STATE_NAME_MAP.get(state, 'unknown')

    def nodeinfo(self):
        return self.conn.getInfo()

    def get_type(self):
        return self.conn.getType()

    def get_xml(self, vmid):
        vm = self.conn.lookupByName(vmid)
        return vm.XMLDesc(0)

    def get_maxVcpus(self, vmid):
        vm = self.conn.lookupByName(vmid)
        return vm.maxVcpus()

    def get_maxMemory(self, vmid):
        vm = self.conn.lookupByName(vmid)
        return vm.maxMemory()

    def getFreeMemory(self):
        return self.conn.getFreeMemory()

    def get_autostart(self, vmid):
        vm = self.conn.lookupByName(vmid)
        return vm.autostart()

    def set_autostart(self, vmid, val):
        vm = self.conn.lookupByName(vmid)
        return vm.setAutostart(val)

    def define_from_xml(self, xml):
        return self.conn.defineXML(xml)