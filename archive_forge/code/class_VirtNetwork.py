from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
class VirtNetwork(object):

    def __init__(self, uri, module):
        self.module = module
        self.uri = uri
        self.conn = LibvirtConnection(self.uri, self.module)

    def get_net(self, entryid):
        return self.conn.find_entry(entryid)

    def list_nets(self, state=None):
        results = []
        for entry in self.conn.find_entry(-1):
            if state:
                if state == self.conn.get_status2(entry):
                    results.append(entry.name())
            else:
                results.append(entry.name())
        return results

    def state(self):
        results = []
        for entry in self.list_nets():
            state_blurb = self.conn.get_status(entry)
            results.append('%s %s' % (entry, state_blurb))
        return results

    def autostart(self, entryid):
        return self.conn.set_autostart(entryid, True)

    def get_autostart(self, entryid):
        return self.conn.get_autostart2(entryid)

    def set_autostart(self, entryid, state):
        return self.conn.set_autostart(entryid, state)

    def create(self, entryid):
        if self.conn.get_status(entryid) == 'active':
            return
        try:
            return self.conn.create(entryid)
        except libvirt.libvirtError as e:
            if e.get_error_code() == libvirt.VIR_ERR_NETWORK_EXIST:
                return None
            raise

    def modify(self, entryid, xml):
        return self.conn.modify(entryid, xml)

    def start(self, entryid):
        return self.create(entryid)

    def stop(self, entryid):
        if self.conn.get_status(entryid) == 'active':
            return self.conn.destroy(entryid)

    def destroy(self, entryid):
        return self.stop(entryid)

    def undefine(self, entryid):
        return self.conn.undefine(entryid)

    def status(self, entryid):
        return self.conn.get_status(entryid)

    def get_xml(self, entryid):
        return self.conn.get_xml(entryid)

    def define(self, entryid, xml):
        return self.conn.define_from_xml(entryid, xml)

    def info(self):
        return self.facts(facts_mode='info')

    def facts(self, name=None, facts_mode='facts'):
        results = dict()
        if name:
            entries = [name]
        else:
            entries = self.list_nets()
        for entry in entries:
            results[entry] = dict()
            results[entry]['autostart'] = self.conn.get_autostart(entry)
            results[entry]['persistent'] = self.conn.get_persistent(entry)
            results[entry]['state'] = self.conn.get_status(entry)
            results[entry]['bridge'] = self.conn.get_bridge(entry)
            results[entry]['uuid'] = self.conn.get_uuid(entry)
            try:
                results[entry]['dhcp_leases'] = self.conn.get_dhcp_leases(entry)
            except AttributeError:
                pass
            try:
                results[entry]['forward_mode'] = self.conn.get_forward(entry)
            except ValueError:
                pass
            try:
                results[entry]['domain'] = self.conn.get_domain(entry)
            except ValueError:
                pass
            try:
                results[entry]['macaddress'] = self.conn.get_macaddress(entry)
            except ValueError:
                pass
        facts = dict()
        if facts_mode == 'facts':
            facts['ansible_facts'] = dict()
            facts['ansible_facts']['ansible_libvirt_networks'] = results
        elif facts_mode == 'info':
            facts['networks'] = results
        return facts