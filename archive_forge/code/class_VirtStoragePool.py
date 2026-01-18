from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class VirtStoragePool(object):

    def __init__(self, uri, module):
        self.module = module
        self.uri = uri
        self.conn = LibvirtConnection(self.uri, self.module)

    def get_pool(self, entryid):
        return self.conn.find_entry(entryid)

    def list_pools(self, state=None):
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
        for entry in self.list_pools():
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
        return self.conn.create(entryid)

    def start(self, entryid):
        return self.conn.create(entryid)

    def stop(self, entryid):
        return self.conn.destroy(entryid)

    def destroy(self, entryid):
        return self.conn.destroy(entryid)

    def undefine(self, entryid):
        return self.conn.undefine(entryid)

    def status(self, entryid):
        return self.conn.get_status(entryid)

    def get_xml(self, entryid):
        return self.conn.get_xml(entryid)

    def define(self, entryid, xml):
        return self.conn.define_from_xml(entryid, xml)

    def build(self, entryid, flags):
        return self.conn.build(entryid, ENTRY_BUILD_FLAGS_MAP.get(flags, 0))

    def delete(self, entryid, flags):
        return self.conn.delete(entryid, ENTRY_DELETE_FLAGS_MAP.get(flags, 0))

    def refresh(self, entryid):
        return self.conn.refresh(entryid)

    def info(self):
        return self.facts(facts_mode='info')

    def facts(self, facts_mode='facts'):
        results = dict()
        for entry in self.list_pools():
            results[entry] = dict()
            if self.conn.find_entry(entry):
                data = self.conn.get_info(entry)
                results[entry] = {'status': ENTRY_STATE_INFO_MAP.get(data[0], 'unknown'), 'size_total': str(data[1]), 'size_used': str(data[2]), 'size_available': str(data[3])}
                results[entry]['autostart'] = self.conn.get_autostart(entry)
                results[entry]['persistent'] = self.conn.get_persistent(entry)
                results[entry]['state'] = self.conn.get_status(entry)
                results[entry]['type'] = self.conn.get_type(entry)
                results[entry]['uuid'] = self.conn.get_uuid(entry)
                if self.conn.find_entry(entry).isActive():
                    results[entry]['volume_count'] = self.conn.get_volume_count(entry)
                    results[entry]['volumes'] = list()
                    for volume in self.conn.get_volume_names(entry):
                        results[entry]['volumes'].append(volume.name())
                else:
                    results[entry]['volume_count'] = -1
                try:
                    results[entry]['path'] = self.conn.get_path(entry)
                except ValueError:
                    pass
                try:
                    results[entry]['host'] = self.conn.get_host(entry)
                except ValueError:
                    pass
                try:
                    results[entry]['source_path'] = self.conn.get_source_path(entry)
                except ValueError:
                    pass
                try:
                    results[entry]['format'] = self.conn.get_format(entry)
                except ValueError:
                    pass
                try:
                    devices = self.conn.get_devices(entry)
                    results[entry]['devices'] = devices
                except ValueError:
                    pass
            else:
                results[entry]['state'] = self.conn.get_status(entry)
        facts = dict()
        if facts_mode == 'facts':
            facts['ansible_facts'] = dict()
            facts['ansible_facts']['ansible_libvirt_pools'] = results
        elif facts_mode == 'info':
            facts['pools'] = results
        return facts