from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
class Virt(object):

    def __init__(self, uri, module):
        self.module = module
        self.uri = uri

    def __get_conn(self):
        self.conn = LibvirtConnection(self.uri, self.module)
        return self.conn

    def get_vm(self, vmid):
        self.__get_conn()
        return self.conn.find_vm(vmid)

    def state(self):
        vms = self.list_vms()
        state = []
        for vm in vms:
            state_blurb = self.conn.get_status(vm)
            state.append('%s %s' % (vm, state_blurb))
        return state

    def info(self):
        vms = self.list_vms()
        info = dict()
        for vm in vms:
            data = self.conn.find_vm(vm).info()
            info[vm] = dict(state=VIRT_STATE_NAME_MAP.get(data[0], 'unknown'), maxMem=str(data[1]), memory=str(data[2]), nrVirtCpu=data[3], cpuTime=str(data[4]), autostart=self.conn.get_autostart(vm))
        return info

    def nodeinfo(self):
        self.__get_conn()
        data = self.conn.nodeinfo()
        info = dict(cpumodel=str(data[0]), phymemory=str(data[1]), cpus=str(data[2]), cpumhz=str(data[3]), numanodes=str(data[4]), sockets=str(data[5]), cpucores=str(data[6]), cputhreads=str(data[7]))
        return info

    def list_vms(self, state=None):
        self.conn = self.__get_conn()
        vms = self.conn.find_vm(-1)
        results = []
        for x in vms:
            try:
                if state:
                    vmstate = self.conn.get_status2(x)
                    if vmstate == state:
                        results.append(x.name())
                else:
                    results.append(x.name())
            except Exception:
                pass
        return results

    def virttype(self):
        return self.__get_conn().get_type()

    def autostart(self, vmid, as_flag):
        self.conn = self.__get_conn()
        if self.conn.get_autostart(vmid) != as_flag:
            self.conn.set_autostart(vmid, as_flag)
            return True
        return False

    def freemem(self):
        self.conn = self.__get_conn()
        return self.conn.getFreeMemory()

    def shutdown(self, vmid):
        """ Make the machine with the given vmid stop running.  Whatever that takes.  """
        self.__get_conn()
        self.conn.shutdown(vmid)
        return 0

    def pause(self, vmid):
        """ Pause the machine with the given vmid.  """
        self.__get_conn()
        return self.conn.suspend(vmid)

    def unpause(self, vmid):
        """ Unpause the machine with the given vmid.  """
        self.__get_conn()
        return self.conn.resume(vmid)

    def create(self, vmid):
        """ Start the machine via the given vmid """
        self.__get_conn()
        return self.conn.create(vmid)

    def start(self, vmid):
        """ Start the machine via the given id/name """
        self.__get_conn()
        return self.conn.create(vmid)

    def destroy(self, vmid):
        """ Pull the virtual power from the virtual domain, giving it virtually no time to virtually shut down.  """
        self.__get_conn()
        return self.conn.destroy(vmid)

    def undefine(self, vmid, flag):
        """ Stop a domain, and then wipe it from the face of the earth.  (delete disk/config file) """
        self.__get_conn()
        return self.conn.undefine(vmid, flag)

    def status(self, vmid):
        """
        Return a state suitable for server consumption.  Aka, codes.py values, not XM output.
        """
        self.__get_conn()
        return self.conn.get_status(vmid)

    def get_xml(self, vmid):
        """
        Receive a Vm id as input
        Return an xml describing vm config returned by a libvirt call
        """
        self.__get_conn()
        return self.conn.get_xml(vmid)

    def get_maxVcpus(self, vmid):
        """
        Gets the max number of VCPUs on a guest
        """
        self.__get_conn()
        return self.conn.get_maxVcpus(vmid)

    def get_max_memory(self, vmid):
        """
        Gets the max memory on a guest
        """
        self.__get_conn()
        return self.conn.get_MaxMemory(vmid)

    def define(self, xml):
        """
        Define a guest with the given xml
        """
        self.__get_conn()
        return self.conn.define_from_xml(xml)