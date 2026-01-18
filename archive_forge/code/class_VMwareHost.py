from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareHost(PyVmomi):
    """Class to manage vCenter connection"""

    def __init__(self, module):
        super(VMwareHost, self).__init__(module)
        self.vcenter = module.params['hostname']
        self.datacenter_name = module.params['datacenter_name']
        self.cluster_name = module.params['cluster_name']
        self.folder_name = module.params['folder']
        self.esxi_hostname = module.params['esxi_hostname']
        self.esxi_username = module.params['esxi_username']
        self.esxi_password = module.params['esxi_password']
        self.state = module.params['state']
        self.esxi_ssl_thumbprint = module.params.get('esxi_ssl_thumbprint', '')
        self.force_connection = module.params.get('force_connection')
        self.fetch_ssl_thumbprint = module.params.get('fetch_ssl_thumbprint')
        self.reconnect_disconnected = module.params.get('reconnect_disconnected')
        self.host_update = self.host = self.cluster = self.folder = self.host_parent_compute_resource = None

    def process_state(self):
        """Check the current state"""
        host_states = {'absent': {'present': self.state_remove_host, 'update': self.state_remove_host, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_exit_unchanged, 'update': self.state_update_host, 'absent': self.state_add_host}, 'add_or_reconnect': {'present': self.state_reconnect_host, 'update': self.state_update_host, 'absent': self.state_add_host}, 'reconnect': {'present': self.state_reconnect_host, 'update': self.state_update_host}, 'disconnected': {'present': self.state_disconnected_host, 'absent': self.state_exit_unchanged}}
        try:
            host_states[self.state][self.check_host_state()]()
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=to_native(runtime_fault.msg))
        except vmodl.MethodFault as method_fault:
            self.module.fail_json(msg=to_native(method_fault.msg))
        except Exception as e:
            self.module.fail_json(msg=to_native(e))

    def check_host_state(self):
        """Check current state"""
        self.host_update = find_hostsystem_by_name(self.content, self.esxi_hostname)
        if self.host_update:
            if self.cluster_name:
                self.host, self.cluster = self.search_cluster(self.datacenter_name, self.cluster_name, self.esxi_hostname)
                if self.host:
                    state = 'present'
                else:
                    state = 'update'
            elif self.folder_name:
                self.folder = self.search_folder(self.folder_name)
                for child in self.folder.childEntity:
                    if not child or not isinstance(child, vim.ComputeResource):
                        continue
                    try:
                        if isinstance(child.host[0], vim.HostSystem) and child.name == self.esxi_hostname:
                            self.host_parent_compute_resource = child
                            self.host = child.host[0]
                            break
                    except IndexError:
                        continue
                if self.host:
                    state = 'present'
                else:
                    state = 'update'
        else:
            state = 'absent'
        return state

    def search_folder(self, folder_name):
        """
            Search folder in vCenter
            Returns: folder object
        """
        search_index = self.content.searchIndex
        folder_obj = search_index.FindByInventoryPath(folder_name)
        if not (folder_obj and isinstance(folder_obj, vim.Folder)):
            self.module.fail_json(msg="Folder '%s' not found" % folder_name)
        return folder_obj

    def search_cluster(self, datacenter_name, cluster_name, esxi_hostname):
        """
            Search cluster in vCenter
            Returns: host and cluster object
        """
        return find_host_by_cluster_datacenter(self.module, self.content, datacenter_name, cluster_name, esxi_hostname)

    def state_exit_unchanged(self):
        """Exit with status message"""
        if not self.host_update:
            result = 'Host already disconnected'
        elif self.reconnect_disconnected and self.host_update.runtime.connectionState == 'disconnected':
            self.state_reconnect_host()
        elif self.folder_name:
            result = "Host already connected to vCenter '%s' in folder '%s'" % (self.vcenter, self.folder_name)
        elif self.cluster_name:
            result = "Host already connected to vCenter '%s' in cluster '%s'" % (self.vcenter, self.cluster_name)
        self.module.exit_json(changed=False, result=str(result))

    def state_add_host(self):
        """Add ESXi host to a cluster of folder in vCenter"""
        changed = True
        result = None
        if self.module.check_mode:
            result = "Host would be connected to vCenter '%s'" % self.vcenter
        else:
            host_connect_spec = self.get_host_connect_spec()
            as_connected = self.params.get('add_connected')
            esxi_license = None
            resource_pool = None
            task = None
            if self.folder_name:
                self.folder = self.search_folder(self.folder_name)
                try:
                    task = self.folder.AddStandaloneHost(spec=host_connect_spec, compResSpec=resource_pool, addConnected=as_connected, license=esxi_license)
                except vim.fault.InvalidLogin as invalid_login:
                    self.module.fail_json(msg='Cannot authenticate with the host : %s' % to_native(invalid_login))
                except vim.fault.HostConnectFault as connect_fault:
                    self.module.fail_json(msg='An error occurred during connect : %s' % to_native(connect_fault))
                except vim.fault.DuplicateName as duplicate_name:
                    self.module.fail_json(msg='The folder already contains a host with the same name : %s' % to_native(duplicate_name))
                except vmodl.fault.InvalidArgument as invalid_argument:
                    self.module.fail_json(msg='An argument was specified incorrectly : %s' % to_native(invalid_argument))
                except vim.fault.AlreadyBeingManaged as already_managed:
                    self.module.fail_json(msg='The host is already being managed by another vCenter server : %s' % to_native(already_managed))
                except vmodl.fault.NotEnoughLicenses as not_enough_licenses:
                    self.module.fail_json(msg='There are not enough licenses to add this host : %s' % to_native(not_enough_licenses))
                except vim.fault.NoHost as no_host:
                    self.module.fail_json(msg='Unable to contact the host : %s' % to_native(no_host))
                except vmodl.fault.NotSupported as not_supported:
                    self.module.fail_json(msg='The folder is not a host folder : %s' % to_native(not_supported))
                except vim.fault.NotSupportedHost as host_not_supported:
                    self.module.fail_json(msg='The host is running a software version that is not supported : %s' % to_native(host_not_supported))
                except vim.fault.AgentInstallFailed as agent_install:
                    self.module.fail_json(msg='Error during vCenter agent installation : %s' % to_native(agent_install))
                except vim.fault.AlreadyConnected as already_connected:
                    self.module.fail_json(msg='The host is already connected to the vCenter server : %s' % to_native(already_connected))
                except vim.fault.SSLVerifyFault as ssl_fault:
                    self.module.fail_json(msg='The host certificate could not be authenticated : %s' % to_native(ssl_fault))
            elif self.cluster_name:
                self.host, self.cluster = self.search_cluster(self.datacenter_name, self.cluster_name, self.esxi_hostname)
                try:
                    task = self.cluster.AddHost_Task(spec=host_connect_spec, asConnected=as_connected, resourcePool=resource_pool, license=esxi_license)
                except vim.fault.InvalidLogin as invalid_login:
                    self.module.fail_json(msg='Cannot authenticate with the host : %s' % to_native(invalid_login))
                except vim.fault.HostConnectFault as connect_fault:
                    self.module.fail_json(msg='An error occurred during connect : %s' % to_native(connect_fault))
                except vim.fault.DuplicateName as duplicate_name:
                    self.module.fail_json(msg='The cluster already contains a host with the same name : %s' % to_native(duplicate_name))
                except vim.fault.AlreadyBeingManaged as already_managed:
                    self.module.fail_json(msg='The host is already being managed by another vCenter server : %s' % to_native(already_managed))
                except vmodl.fault.NotEnoughLicenses as not_enough_licenses:
                    self.module.fail_json(msg='There are not enough licenses to add this host : %s' % to_native(not_enough_licenses))
                except vim.fault.NoHost as no_host:
                    self.module.fail_json(msg='Unable to contact the host : %s' % to_native(no_host))
                except vim.fault.NotSupportedHost as host_not_supported:
                    self.module.fail_json(msg='The host is running a software version that is not supported; It may still be possible to add the host as a stand-alone host : %s' % to_native(host_not_supported))
                except vim.fault.TooManyHosts as too_many_hosts:
                    self.module.fail_json(msg='No additional hosts can be added to the cluster : %s' % to_native(too_many_hosts))
                except vim.fault.AgentInstallFailed as agent_install:
                    self.module.fail_json(msg='Error during vCenter agent installation : %s' % to_native(agent_install))
                except vim.fault.AlreadyConnected as already_connected:
                    self.module.fail_json(msg='The host is already connected to the vCenter server : %s' % to_native(already_connected))
                except vim.fault.SSLVerifyFault as ssl_fault:
                    self.module.fail_json(msg='The host certificate could not be authenticated : %s' % to_native(ssl_fault))
            try:
                changed, result = wait_for_task(task)
                result = "Host connected to vCenter '%s'" % self.vcenter
            except TaskError as task_error:
                self.module.fail_json(msg="Failed to add host to vCenter '%s' : %s" % (self.vcenter, to_native(task_error)))
        self.module.exit_json(changed=changed, result=result)

    def get_host_connect_spec(self):
        """
        Function to return Host connection specification
        Returns: host connection specification
        """
        if self.fetch_ssl_thumbprint and self.esxi_ssl_thumbprint == '':
            sslThumbprint = self.get_cert_fingerprint(self.esxi_hostname, self.module.params['port'], self.module.params['proxy_host'], self.module.params['proxy_port'])
        else:
            sslThumbprint = self.esxi_ssl_thumbprint
        host_connect_spec = vim.host.ConnectSpec()
        host_connect_spec.sslThumbprint = sslThumbprint
        host_connect_spec.hostName = self.esxi_hostname
        host_connect_spec.userName = self.esxi_username
        host_connect_spec.password = self.esxi_password
        host_connect_spec.force = self.force_connection
        return host_connect_spec

    def state_reconnect_host(self):
        """Reconnect host to vCenter"""
        changed = True
        result = None
        if self.module.check_mode:
            result = "Host would be reconnected to vCenter '%s'" % self.vcenter
        else:
            self.reconnect_host(self.host)
            result = "Host reconnected to vCenter '%s'" % self.vcenter
        self.module.exit_json(changed=changed, result=str(result))

    def reconnect_host(self, host_object):
        """Reconnect host to vCenter"""
        reconnecthost_args = {}
        reconnecthost_args['reconnectSpec'] = vim.HostSystem.ReconnectSpec()
        reconnecthost_args['reconnectSpec'].syncState = True
        if self.esxi_username and self.esxi_password:
            reconnecthost_args['cnxSpec'] = self.get_host_connect_spec()
        try:
            task = host_object.ReconnectHost_Task(**reconnecthost_args)
        except vim.fault.InvalidLogin as invalid_login:
            self.module.fail_json(msg='Cannot authenticate with the host : %s' % to_native(invalid_login))
        except vim.fault.InvalidState as invalid_state:
            self.module.fail_json(msg='The host is not disconnected : %s' % to_native(invalid_state))
        except vim.fault.InvalidName as invalid_name:
            self.module.fail_json(msg='The host name is invalid : %s' % to_native(invalid_name))
        except vim.fault.HostConnectFault as connect_fault:
            self.module.fail_json(msg='An error occurred during reconnect : %s' % to_native(connect_fault))
        except vmodl.fault.NotSupported as not_supported:
            self.module.fail_json(msg='No host can be added to this group : %s' % to_native(not_supported))
        except vim.fault.AlreadyBeingManaged as already_managed:
            self.module.fail_json(msg='The host is already being managed by another vCenter server : %s' % to_native(already_managed))
        except vmodl.fault.NotEnoughLicenses as not_enough_licenses:
            self.module.fail_json(msg='There are not enough licenses to add this host : %s' % to_native(not_enough_licenses))
        except vim.fault.NoHost as no_host:
            self.module.fail_json(msg='Unable to contact the host : %s' % to_native(no_host))
        except vim.fault.NotSupportedHost as host_not_supported:
            self.module.fail_json(msg='The host is running a software version that is not supported : %s' % to_native(host_not_supported))
        except vim.fault.SSLVerifyFault as ssl_fault:
            self.module.fail_json(msg='The host certificate could not be authenticated : %s' % to_native(ssl_fault))
        try:
            changed, result = wait_for_task(task)
        except TaskError as task_error:
            self.module.fail_json(msg="Failed to reconnect host to vCenter '%s' due to %s" % (self.vcenter, to_native(task_error)))

    def state_remove_host(self):
        """Remove host from vCenter"""
        changed = True
        result = None
        if self.module.check_mode:
            result = "Host would be removed from vCenter '%s'" % self.vcenter
        else:
            parent_type = self.get_parent_type(self.host_update)
            if parent_type == 'cluster':
                self.put_host_in_maintenance_mode(self.host_update)
            try:
                if self.folder_name:
                    task = self.host_parent_compute_resource.Destroy_Task()
                elif self.cluster_name:
                    task = self.host.Destroy_Task()
            except vim.fault.VimFault as vim_fault:
                self.module.fail_json(msg=vim_fault)
            try:
                changed, result = wait_for_task(task)
                result = "Host removed from vCenter '%s'" % self.vcenter
            except TaskError as task_error:
                self.module.fail_json(msg="Failed to remove the host from vCenter '%s' : %s" % (self.vcenter, to_native(task_error)))
        self.module.exit_json(changed=changed, result=str(result))

    def put_host_in_maintenance_mode(self, host_object):
        """Put host in maintenance mode, if not already"""
        if not host_object.runtime.inMaintenanceMode:
            try:
                try:
                    maintenance_mode_task = host_object.EnterMaintenanceMode_Task(300, True, None)
                except vim.fault.InvalidState as invalid_state:
                    self.module.fail_json(msg='The host is already in maintenance mode : %s' % to_native(invalid_state))
                except vim.fault.Timedout as timed_out:
                    self.module.fail_json(msg='The maintenance mode operation timed out : %s' % to_native(timed_out))
                except vim.fault.Timedout as timed_out:
                    self.module.fail_json(msg='The maintenance mode operation was canceled : %s' % to_native(timed_out))
                wait_for_task(maintenance_mode_task)
            except TaskError as task_err:
                self.module.fail_json(msg='Failed to put the host in maintenance mode : %s' % to_native(task_err))

    def get_parent_type(self, host_object):
        """
            Get the type of the parent object
            Returns: string with 'folder' or 'cluster'
        """
        object_type = None
        if isinstance(host_object.parent, vim.ClusterComputeResource):
            object_type = 'cluster'
        elif isinstance(host_object.parent, vim.ComputeResource):
            object_type = 'folder'
        return object_type

    def state_update_host(self):
        """Move host to a cluster or a folder, or vice versa"""
        changed = True
        result = None
        reconnect = False
        if self.reconnect_disconnected and self.host_update.runtime.connectionState == 'disconnected':
            reconnect = True
        parent_type = self.get_parent_type(self.host_update)
        if self.folder_name:
            if self.module.check_mode:
                if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                    result = "Host would be reconnected and moved to folder '%s'" % self.folder_name
                else:
                    result = "Host would be moved to folder '%s'" % self.folder_name
            else:
                if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                    self.reconnect_host(self.host_update)
                try:
                    try:
                        if parent_type == 'folder':
                            task = self.folder.MoveIntoFolder_Task([self.host_update.parent])
                        elif parent_type == 'cluster':
                            self.put_host_in_maintenance_mode(self.host_update)
                            task = self.folder.MoveIntoFolder_Task([self.host_update])
                    except vim.fault.DuplicateName as duplicate_name:
                        self.module.fail_json(msg='The folder already contains an object with the specified name : %s' % to_native(duplicate_name))
                    except vim.fault.InvalidFolder as invalid_folder:
                        self.module.fail_json(msg='The parent of this folder is in the list of objects : %s' % to_native(invalid_folder))
                    except vim.fault.InvalidState as invalid_state:
                        self.module.fail_json(msg='Failed to move host, this can be due to either of following : 1. The host is not part of the same datacenter, 2. The host is not in maintenance mode : %s' % to_native(invalid_state))
                    except vmodl.fault.NotSupported as not_supported:
                        self.module.fail_json(msg='The target folder is not a host folder : %s' % to_native(not_supported))
                    except vim.fault.DisallowedOperationOnFailoverHost as failover_host:
                        self.module.fail_json(msg='The host is configured as a failover host : %s' % to_native(failover_host))
                    except vim.fault.VmAlreadyExistsInDatacenter as already_exists:
                        self.module.fail_json(msg="The host's virtual machines are already registered to a host in the destination datacenter : %s" % to_native(already_exists))
                    changed, result = wait_for_task(task)
                except TaskError as task_error_exception:
                    task_error = task_error_exception.args[0]
                    self.module.fail_json(msg='Failed to move host %s to folder %s due to %s' % (self.esxi_hostname, self.folder_name, to_native(task_error)))
                if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                    result = "Host reconnected and moved to folder '%s'" % self.folder_name
                else:
                    result = "Host moved to folder '%s'" % self.folder_name
        elif self.cluster_name:
            if self.module.check_mode:
                result = "Host would be moved to cluster '%s'" % self.cluster_name
            else:
                if parent_type == 'cluster':
                    self.put_host_in_maintenance_mode(self.host_update)
                resource_pool = None
                try:
                    try:
                        task = self.cluster.MoveHostInto_Task(host=self.host_update, resourcePool=resource_pool)
                    except vim.fault.TooManyHosts as too_many_hosts:
                        self.module.fail_json(msg='No additional hosts can be added to the cluster : %s' % to_native(too_many_hosts))
                    except vim.fault.InvalidState as invalid_state:
                        self.module.fail_json(msg='The host is already part of a cluster and is not in maintenance mode : %s' % to_native(invalid_state))
                    except vmodl.fault.InvalidArgument as invalid_argument:
                        self.module.fail_json(msg='Failed to move host, this can be due to either of following : 1. The host is is not a part of the same datacenter as the cluster, 2. The source and destination clusters are the same : %s' % to_native(invalid_argument))
                    changed, result = wait_for_task(task)
                except TaskError as task_error_exception:
                    task_error = task_error_exception.args[0]
                    self.module.fail_json(msg="Failed to move host to cluster '%s' due to : %s" % (self.cluster_name, to_native(task_error)))
                if reconnect or self.state == 'add_or_reconnect' or self.state == 'reconnect':
                    result = "Host reconnected and moved to cluster '%s'" % self.cluster_name
                else:
                    result = "Host moved to cluster '%s'" % self.cluster_name
        self.module.exit_json(changed=changed, msg=str(result))

    def state_disconnected_host(self):
        """Disconnect host to vCenter"""
        changed = True
        result = None
        if self.module.check_mode:
            if self.host.runtime.connectionState == 'disconnected':
                result = 'Host already disconnected'
                changed = False
            else:
                result = "Host would be disconnected host from vCenter '%s'" % self.vcenter
        elif self.host.runtime.connectionState == 'disconnected':
            changed = False
            result = 'Host already disconnected'
        else:
            self.disconnect_host(self.host)
            result = "Host disconnected from vCenter '%s'" % self.vcenter
        self.module.exit_json(changed=changed, result=to_native(result))

    def disconnect_host(self, host_object):
        """Disconnect host to vCenter"""
        try:
            task = host_object.DisconnectHost_Task()
        except Exception as e:
            self.module.fail_json(msg='Failed to disconnect host from vCenter: %s' % to_native(e))
        try:
            changed, result = wait_for_task(task)
        except TaskError as task_error:
            self.module.fail_json(msg="Failed to disconnect host from vCenter '%s' due to %s" % (self.vcenter, to_native(task_error)))