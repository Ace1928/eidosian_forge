from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
class VmwareDrsGroupInfoManager(PyVmomi):

    def __init__(self, module, datacenter_name, cluster_name=None):
        """
        Doctring: Init
        """
        super(VmwareDrsGroupInfoManager, self).__init__(module)
        self.__datacenter_name = datacenter_name
        self.__datacenter_obj = None
        self.__cluster_name = cluster_name
        self.__cluster_obj = None
        self.__result = dict()
        if self.__datacenter_name:
            self.__datacenter_obj = find_datacenter_by_name(self.content, datacenter_name=self.__datacenter_name)
            self.cluster_obj_list = []
            if self.__datacenter_obj:
                folder = self.__datacenter_obj.hostFolder
                self.cluster_obj_list = get_all_objs(self.content, [vim.ClusterComputeResource], folder)
            else:
                raise Exception("Datacenter '%s' not found" % self.__datacenter_name)
        if self.__cluster_name:
            self.__cluster_obj = self.find_cluster_by_name(cluster_name=self.__cluster_name)
            if self.__cluster_obj is None:
                raise Exception("Cluster '%s' not found" % self.__cluster_name)
            else:
                self.cluster_obj_list = [self.__cluster_obj]

    def get_result(self):
        """
        Docstring
        """
        return self.__result

    def __set_result(self, result):
        """
        Sets result
        Args:
            result: drs group result list

        Returns: None

        """
        self.__result = result

    def __get_all_from_group(self, group_obj, host_group=False):
        """
        Return all VM / Host names using given group
        Args:
            group_obj: Group object
            host_group: True if we want only host name from group

        Returns: List of VM / Host names belonging to given group object

        """
        obj_name_list = []
        if not all([group_obj]):
            return obj_name_list
        if not host_group and isinstance(group_obj, vim.cluster.VmGroup):
            obj_name_list = [vm.name for vm in group_obj.vm]
        elif host_group and isinstance(group_obj, vim.cluster.HostGroup):
            obj_name_list = [host.name for host in group_obj.host]
        return obj_name_list

    def __normalize_group_data(self, group_obj):
        """
        Return human readable group spec
        Args:
            group_obj: Group object

        Returns: Dictionary with DRS groups

        """
        if not all([group_obj]):
            return {}
        if hasattr(group_obj, 'host'):
            return dict(group_name=group_obj.name, hosts=self.__get_all_from_group(group_obj=group_obj, host_group=True), type='host')
        else:
            return dict(group_name=group_obj.name, vms=self.__get_all_from_group(group_obj=group_obj), type='vm')

    def gather_info(self):
        """
        Gather DRS group information about given cluster
        Returns: Dictionary of clusters with DRS groups

        """
        cluster_group_info = dict()
        for cluster_obj in self.cluster_obj_list:
            cluster_group_info[cluster_obj.name] = []
            for drs_group in cluster_obj.configurationEx.group:
                cluster_group_info[cluster_obj.name].append(self.__normalize_group_data(drs_group))
        self.__set_result(cluster_group_info)