from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping
from ansible import constants as C
from ansible.template import Templar, AnsibleUndefined
class HostVars(Mapping):
    """ A special view of vars_cache that adds values from the inventory when needed. """

    def __init__(self, inventory, variable_manager, loader):
        self._inventory = inventory
        self._loader = loader
        self._variable_manager = variable_manager
        variable_manager._hostvars = self

    def set_variable_manager(self, variable_manager):
        self._variable_manager = variable_manager
        variable_manager._hostvars = self

    def set_inventory(self, inventory):
        self._inventory = inventory

    def _find_host(self, host_name):
        return self._inventory.get_host(host_name)

    def raw_get(self, host_name):
        """
        Similar to __getitem__, however the returned data is not run through
        the templating engine to expand variables in the hostvars.
        """
        host = self._find_host(host_name)
        if host is None:
            return AnsibleUndefined(name="hostvars['%s']" % host_name)
        return self._variable_manager.get_vars(host=host, include_hostvars=False)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._variable_manager._loader is None:
            self._variable_manager._loader = self._loader
        if self._variable_manager._hostvars is None:
            self._variable_manager._hostvars = self

    def __getitem__(self, host_name):
        data = self.raw_get(host_name)
        if isinstance(data, AnsibleUndefined):
            return data
        return HostVarsVars(data, loader=self._loader)

    def set_host_variable(self, host, varname, value):
        self._variable_manager.set_host_variable(host, varname, value)

    def set_nonpersistent_facts(self, host, facts):
        self._variable_manager.set_nonpersistent_facts(host, facts)

    def set_host_facts(self, host, facts):
        self._variable_manager.set_host_facts(host, facts)

    def __contains__(self, host_name):
        return self._find_host(host_name) is not None

    def __iter__(self):
        for host in self._inventory.hosts:
            yield host

    def __len__(self):
        return len(self._inventory.hosts)

    def __repr__(self):
        out = {}
        for host in self._inventory.hosts:
            out[host] = self.get(host)
        return repr(out)

    def __deepcopy__(self, memo):
        return self