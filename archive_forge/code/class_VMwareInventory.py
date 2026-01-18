from __future__ import absolute_import, division, print_function
import json
import logging
import optparse
import os
import ssl
import sys
import time
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.module_utils.six import integer_types, text_type, string_types
from ansible.module_utils.six.moves import configparser
from psphere.client import Client
from psphere.errors import ObjectNotFoundError
from psphere.managedobjects import HostSystem, VirtualMachine, ManagedObject, ClusterComputeResource
from suds.sudsobject import Object as SudsObject
class VMwareInventory(object):

    def __init__(self, guests_only=None):
        self.config = configparser.ConfigParser()
        if os.environ.get('VMWARE_INI', ''):
            config_files = [os.environ['VMWARE_INI']]
        else:
            config_files = [os.path.abspath(sys.argv[0]).rstrip('.py') + '.ini', 'vmware.ini']
        for config_file in config_files:
            if os.path.exists(config_file):
                self.config.read(config_file)
                break
        if guests_only is not None:
            self.guests_only = guests_only
        elif self.config.has_option('defaults', 'guests_only'):
            self.guests_only = self.config.getboolean('defaults', 'guests_only')
        else:
            self.guests_only = True
        auth_host = os.environ.get('VMWARE_HOST')
        if not auth_host and self.config.has_option('auth', 'host'):
            auth_host = self.config.get('auth', 'host')
        auth_user = os.environ.get('VMWARE_USER')
        if not auth_user and self.config.has_option('auth', 'user'):
            auth_user = self.config.get('auth', 'user')
        auth_password = os.environ.get('VMWARE_PASSWORD')
        if not auth_password and self.config.has_option('auth', 'password'):
            auth_password = self.config.get('auth', 'password')
        sslcheck = os.environ.get('VMWARE_SSLCHECK')
        if not sslcheck and self.config.has_option('auth', 'sslcheck'):
            sslcheck = self.config.get('auth', 'sslcheck')
        if not sslcheck:
            sslcheck = True
        elif sslcheck.lower() in ['no', 'false']:
            sslcheck = False
        else:
            sslcheck = True
        self.filter_clusters = os.environ.get('VMWARE_CLUSTERS')
        if not self.filter_clusters and self.config.has_option('defaults', 'clusters'):
            self.filter_clusters = self.config.get('defaults', 'clusters')
        if self.filter_clusters:
            self.filter_clusters = [x.strip() for x in self.filter_clusters.split(',') if x.strip()]
        if not sslcheck:
            if hasattr(ssl, '_create_unverified_context'):
                ssl._create_default_https_context = ssl._create_unverified_context
        self.client = Client(auth_host, auth_user, auth_password)

    def _put_cache(self, name, value):
        """
        Saves the value to cache with the name given.
        """
        if self.config.has_option('defaults', 'cache_dir'):
            cache_dir = os.path.expanduser(self.config.get('defaults', 'cache_dir'))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_file = os.path.join(cache_dir, name)
            with open(cache_file, 'w') as cache:
                json.dump(value, cache)

    def _get_cache(self, name, default=None):
        """
        Retrieves the value from cache for the given name.
        """
        if self.config.has_option('defaults', 'cache_dir'):
            cache_dir = self.config.get('defaults', 'cache_dir')
            cache_file = os.path.join(cache_dir, name)
            if os.path.exists(cache_file):
                if self.config.has_option('defaults', 'cache_max_age'):
                    cache_max_age = self.config.getint('defaults', 'cache_max_age')
                else:
                    cache_max_age = 0
                cache_stat = os.stat(cache_file)
                if cache_stat.st_mtime + cache_max_age >= time.time():
                    with open(cache_file) as cache:
                        return json.load(cache)
        return default

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten nested dicts by combining keys with a separator.  Lists with
        only string items are included as is; any other lists are discarded.
        """
        items = []
        for k, v in d.items():
            if k.startswith('_'):
                continue
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                if all((isinstance(x, string_types) for x in v)):
                    items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    def _get_obj_info(self, obj, depth=99, seen=None):
        """
        Recursively build a data structure for the given pSphere object (depth
        only applies to ManagedObject instances).
        """
        seen = seen or set()
        if isinstance(obj, ManagedObject):
            try:
                obj_unicode = text_type(getattr(obj, 'name'))
            except AttributeError:
                obj_unicode = ()
            if obj in seen:
                return obj_unicode
            seen.add(obj)
            if depth <= 0:
                return obj_unicode
            d = {}
            for attr in dir(obj):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(obj, attr)
                    obj_info = self._get_obj_info(val, depth - 1, seen)
                    if obj_info != ():
                        d[attr] = obj_info
                except Exception as e:
                    pass
            return d
        elif isinstance(obj, SudsObject):
            d = {}
            for key, val in iter(obj):
                obj_info = self._get_obj_info(val, depth, seen)
                if obj_info != ():
                    d[key] = obj_info
            return d
        elif isinstance(obj, (list, tuple)):
            l = []
            for val in iter(obj):
                obj_info = self._get_obj_info(val, depth, seen)
                if obj_info != ():
                    l.append(obj_info)
            return l
        elif isinstance(obj, (type(None), bool, float) + string_types + integer_types):
            return obj
        else:
            return ()

    def _get_host_info(self, host, prefix='vmware'):
        """
        Return a flattened dict with info about the given host system.
        """
        host_info = {'name': host.name}
        for attr in ('datastore', 'network', 'vm'):
            try:
                value = getattr(host, attr)
                host_info['%ss' % attr] = self._get_obj_info(value, depth=0)
            except AttributeError:
                host_info['%ss' % attr] = []
        for k, v in self._get_obj_info(host.summary, depth=0).items():
            if isinstance(v, MutableMapping):
                for k2, v2 in v.items():
                    host_info[k2] = v2
            elif k != 'host':
                host_info[k] = v
        try:
            host_info['ipAddress'] = host.config.network.vnic[0].spec.ip.ipAddress
        except Exception as e:
            print(e, file=sys.stderr)
        host_info = self._flatten_dict(host_info, prefix)
        if '%s_ipAddress' % prefix in host_info:
            host_info['ansible_ssh_host'] = host_info['%s_ipAddress' % prefix]
        return host_info

    def _get_vm_info(self, vm, prefix='vmware'):
        """
        Return a flattened dict with info about the given virtual machine.
        """
        vm_info = {'name': vm.name}
        for attr in ('datastore', 'network'):
            try:
                value = getattr(vm, attr)
                vm_info['%ss' % attr] = self._get_obj_info(value, depth=0)
            except AttributeError:
                vm_info['%ss' % attr] = []
        try:
            vm_info['resourcePool'] = self._get_obj_info(vm.resourcePool, depth=0)
        except AttributeError:
            vm_info['resourcePool'] = ''
        try:
            vm_info['guestState'] = vm.guest.guestState
        except AttributeError:
            vm_info['guestState'] = ''
        for k, v in self._get_obj_info(vm.summary, depth=0).items():
            if isinstance(v, MutableMapping):
                for k2, v2 in v.items():
                    if k2 == 'host':
                        k2 = 'hostSystem'
                    vm_info[k2] = v2
            elif k != 'vm':
                vm_info[k] = v
        vm_info = self._flatten_dict(vm_info, prefix)
        if '%s_ipAddress' % prefix in vm_info:
            vm_info['ansible_ssh_host'] = vm_info['%s_ipAddress' % prefix]
        return vm_info

    def _add_host(self, inv, parent_group, host_name):
        """
        Add the host to the parent group in the given inventory.
        """
        p_group = inv.setdefault(parent_group, [])
        if isinstance(p_group, dict):
            group_hosts = p_group.setdefault('hosts', [])
        else:
            group_hosts = p_group
        if host_name not in group_hosts:
            group_hosts.append(host_name)

    def _add_child(self, inv, parent_group, child_group):
        """
        Add a child group to a parent group in the given inventory.
        """
        if parent_group != 'all':
            p_group = inv.setdefault(parent_group, {})
            if not isinstance(p_group, dict):
                inv[parent_group] = {'hosts': p_group}
                p_group = inv[parent_group]
            group_children = p_group.setdefault('children', [])
            if child_group not in group_children:
                group_children.append(child_group)
        inv.setdefault(child_group, [])

    def get_inventory(self, meta_hostvars=True):
        """
        Reads the inventory from cache or VMware API via pSphere.
        """
        if self.guests_only:
            cache_name = '__inventory_guests__'
        else:
            cache_name = '__inventory_all__'
        inv = self._get_cache(cache_name, None)
        if inv is not None:
            return inv
        inv = {'all': {'hosts': []}}
        if meta_hostvars:
            inv['_meta'] = {'hostvars': {}}
        default_group = os.path.basename(sys.argv[0]).rstrip('.py')
        if not self.guests_only:
            if self.config.has_option('defaults', 'hw_group'):
                hw_group = self.config.get('defaults', 'hw_group')
            else:
                hw_group = default_group + '_hw'
        if self.config.has_option('defaults', 'vm_group'):
            vm_group = self.config.get('defaults', 'vm_group')
        else:
            vm_group = default_group + '_vm'
        if self.config.has_option('defaults', 'prefix_filter'):
            prefix_filter = self.config.get('defaults', 'prefix_filter')
        else:
            prefix_filter = None
        if self.filter_clusters:
            hosts = []
            for cluster in ClusterComputeResource.all(self.client):
                if cluster.name in self.filter_clusters:
                    for host in cluster.host:
                        hosts.append(host)
        else:
            hosts = HostSystem.all(self.client)
        for host in hosts:
            if not self.guests_only:
                self._add_host(inv, 'all', host.name)
                self._add_host(inv, hw_group, host.name)
                host_info = self._get_host_info(host)
                if meta_hostvars:
                    inv['_meta']['hostvars'][host.name] = host_info
                self._put_cache(host.name, host_info)
            for vm in host.vm:
                if prefix_filter:
                    if vm.name.startswith(prefix_filter):
                        continue
                self._add_host(inv, 'all', vm.name)
                self._add_host(inv, vm_group, vm.name)
                vm_info = self._get_vm_info(vm)
                if meta_hostvars:
                    inv['_meta']['hostvars'][vm.name] = vm_info
                self._put_cache(vm.name, vm_info)
                vm_resourcePool = vm_info.get('vmware_resourcePool', None)
                if vm_resourcePool:
                    self._add_child(inv, vm_group, 'resource_pools')
                    self._add_child(inv, 'resource_pools', vm_resourcePool)
                    self._add_host(inv, vm_resourcePool, vm.name)
                for vm_datastore in vm_info.get('vmware_datastores', []):
                    self._add_child(inv, vm_group, 'datastores')
                    self._add_child(inv, 'datastores', vm_datastore)
                    self._add_host(inv, vm_datastore, vm.name)
                for vm_network in vm_info.get('vmware_networks', []):
                    self._add_child(inv, vm_group, 'networks')
                    self._add_child(inv, 'networks', vm_network)
                    self._add_host(inv, vm_network, vm.name)
                vm_guestId = vm_info.get('vmware_guestId', None)
                if vm_guestId:
                    self._add_child(inv, vm_group, 'guests')
                    self._add_child(inv, 'guests', vm_guestId)
                    self._add_host(inv, vm_guestId, vm.name)
                vm_template = vm_info.get('vmware_template', False)
                if vm_template:
                    self._add_child(inv, vm_group, 'templates')
                    self._add_host(inv, 'templates', vm.name)
        self._put_cache(cache_name, inv)
        return inv

    def get_host(self, hostname):
        """
        Read info about a specific host or VM from cache or VMware API.
        """
        inv = self._get_cache(hostname, None)
        if inv is not None:
            return inv
        if not self.guests_only:
            try:
                host = HostSystem.get(self.client, name=hostname)
                inv = self._get_host_info(host)
            except ObjectNotFoundError:
                pass
        if inv is None:
            try:
                vm = VirtualMachine.get(self.client, name=hostname)
                inv = self._get_vm_info(vm)
            except ObjectNotFoundError:
                pass
        if inv is not None:
            self._put_cache(hostname, inv)
        return inv or {}