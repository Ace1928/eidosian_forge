from time import sleep
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class ElastiCacheManager:
    """Handles elasticache creation and destruction"""
    EXIST_STATUSES = ['available', 'creating', 'rebooting', 'modifying']

    def __init__(self, module, name, engine, cache_engine_version, node_type, num_nodes, cache_port, cache_parameter_group, cache_subnet_group, cache_security_groups, security_group_ids, zone, wait, hard_modify):
        self.module = module
        self.name = name
        self.engine = engine.lower()
        self.cache_engine_version = cache_engine_version
        self.node_type = node_type
        self.num_nodes = num_nodes
        self.cache_port = cache_port
        self.cache_parameter_group = cache_parameter_group
        self.cache_subnet_group = cache_subnet_group
        self.cache_security_groups = cache_security_groups
        self.security_group_ids = security_group_ids
        self.zone = zone
        self.wait = wait
        self.hard_modify = hard_modify
        self.changed = False
        self.data = None
        self.status = 'gone'
        self.conn = self._get_elasticache_connection()
        self._refresh_data()

    def ensure_present(self):
        """Ensure cache cluster exists or create it if not"""
        if self.exists():
            self.sync()
        else:
            self.create()

    def ensure_absent(self):
        """Ensure cache cluster is gone or delete it if not"""
        self.delete()

    def ensure_rebooted(self):
        """Ensure cache cluster is gone or delete it if not"""
        self.reboot()

    def exists(self):
        """Check if cache cluster exists"""
        return self.status in self.EXIST_STATUSES

    def create(self):
        """Create an ElastiCache cluster"""
        if self.status == 'available':
            return
        if self.status in ['creating', 'rebooting', 'modifying']:
            if self.wait:
                self._wait_for_status('available')
            return
        if self.status == 'deleting':
            if self.wait:
                self._wait_for_status('gone')
            else:
                self.module.fail_json(msg=f"'{self.name}' is currently deleting. Cannot create.")
        kwargs = dict(CacheClusterId=self.name, NumCacheNodes=self.num_nodes, CacheNodeType=self.node_type, Engine=self.engine, EngineVersion=self.cache_engine_version, CacheSecurityGroupNames=self.cache_security_groups, SecurityGroupIds=self.security_group_ids, CacheParameterGroupName=self.cache_parameter_group, CacheSubnetGroupName=self.cache_subnet_group)
        if self.cache_port is not None:
            kwargs['Port'] = self.cache_port
        if self.zone is not None:
            kwargs['PreferredAvailabilityZone'] = self.zone
        try:
            self.conn.create_cache_cluster(**kwargs)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Failed to create cache cluster')
        self._refresh_data()
        self.changed = True
        if self.wait:
            self._wait_for_status('available')
        return True

    def delete(self):
        """Destroy an ElastiCache cluster"""
        if self.status == 'gone':
            return
        if self.status == 'deleting':
            if self.wait:
                self._wait_for_status('gone')
            return
        if self.status in ['creating', 'rebooting', 'modifying']:
            if self.wait:
                self._wait_for_status('available')
            else:
                self.module.fail_json(msg=f"'{self.name}' is currently {self.status}. Cannot delete.")
        try:
            response = self.conn.delete_cache_cluster(CacheClusterId=self.name)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Failed to delete cache cluster')
        cache_cluster_data = response['CacheCluster']
        self._refresh_data(cache_cluster_data)
        self.changed = True
        if self.wait:
            self._wait_for_status('gone')

    def sync(self):
        """Sync settings to cluster if required"""
        if not self.exists():
            self.module.fail_json(msg=f"'{self.name}' is {self.status}. Cannot sync.")
        if self.status in ['creating', 'rebooting', 'modifying']:
            if self.wait:
                self._wait_for_status('available')
            else:
                return
        if self._requires_destroy_and_create():
            if not self.hard_modify:
                self.module.fail_json(msg=f"'{self.name}' requires destructive modification. 'hard_modify' must be set to true to proceed.")
            if not self.wait:
                self.module.fail_json(msg=f"'{self.name}' requires destructive modification. 'wait' must be set to true to proceed.")
            self.delete()
            self.create()
            return
        if self._requires_modification():
            self.modify()

    def modify(self):
        """Modify the cache cluster. Note it's only possible to modify a few select options."""
        nodes_to_remove = self._get_nodes_to_remove()
        try:
            self.conn.modify_cache_cluster(CacheClusterId=self.name, NumCacheNodes=self.num_nodes, CacheNodeIdsToRemove=nodes_to_remove, CacheSecurityGroupNames=self.cache_security_groups, CacheParameterGroupName=self.cache_parameter_group, SecurityGroupIds=self.security_group_ids, ApplyImmediately=True, EngineVersion=self.cache_engine_version)
        except botocore.exceptions.ClientError as e:
            self.module.fail_json_aws(e, msg='Failed to modify cache cluster')
        self._refresh_data()
        self.changed = True
        if self.wait:
            self._wait_for_status('available')

    def reboot(self):
        """Reboot the cache cluster"""
        if not self.exists():
            self.module.fail_json(msg=f"'{self.name}' is {self.status}. Cannot reboot.")
        if self.status == 'rebooting':
            return
        if self.status in ['creating', 'modifying']:
            if self.wait:
                self._wait_for_status('available')
            else:
                self.module.fail_json(msg=f"'{self.name}' is currently {self.status}. Cannot reboot.")
        cache_node_ids = [cn['CacheNodeId'] for cn in self.data['CacheNodes']]
        try:
            self.conn.reboot_cache_cluster(CacheClusterId=self.name, CacheNodeIdsToReboot=cache_node_ids)
        except botocore.exceptions.ClientError as e:
            self.module.fail_json_aws(e, msg='Failed to reboot cache cluster')
        self._refresh_data()
        self.changed = True
        if self.wait:
            self._wait_for_status('available')

    def get_info(self):
        """Return basic info about the cache cluster"""
        info = {'name': self.name, 'status': self.status}
        if self.data:
            info['data'] = self.data
        return info

    def _wait_for_status(self, awaited_status):
        """Wait for status to change from present status to awaited_status"""
        status_map = {'creating': 'available', 'rebooting': 'available', 'modifying': 'available', 'deleting': 'gone'}
        if self.status == awaited_status:
            return
        if status_map[self.status] != awaited_status:
            self.module.fail_json(msg=f"Invalid awaited status. '{self.status}' cannot transition to '{awaited_status}'")
        if awaited_status not in set(status_map.values()):
            self.module.fail_json(msg=f"'{awaited_status}' is not a valid awaited status.")
        while True:
            sleep(1)
            self._refresh_data()
            if self.status == awaited_status:
                break

    def _requires_modification(self):
        """Check if cluster requires (nondestructive) modification"""
        modifiable_data = {'NumCacheNodes': self.num_nodes, 'EngineVersion': self.cache_engine_version}
        for key, value in modifiable_data.items():
            if value is not None and value and (self.data[key] != value):
                return True
        cache_security_groups = []
        for sg in self.data['CacheSecurityGroups']:
            cache_security_groups.append(sg['CacheSecurityGroupName'])
        if set(cache_security_groups) != set(self.cache_security_groups):
            return True
        if self.security_group_ids:
            vpc_security_groups = []
            security_groups = self.data.get('SecurityGroups', [])
            for sg in security_groups:
                vpc_security_groups.append(sg['SecurityGroupId'])
            if set(vpc_security_groups) != set(self.security_group_ids):
                return True
        return False

    def _requires_destroy_and_create(self):
        """
        Check whether a destroy and create is required to synchronize cluster.
        """
        unmodifiable_data = {'node_type': self.data['CacheNodeType'], 'engine': self.data['Engine'], 'cache_port': self._get_port()}
        if self.zone is not None:
            unmodifiable_data['zone'] = self.data['PreferredAvailabilityZone']
        for key, value in unmodifiable_data.items():
            if getattr(self, key) is not None and getattr(self, key) != value:
                return True
        return False

    def _get_elasticache_connection(self):
        """Get an elasticache connection"""
        try:
            return self.module.client('elasticache')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Failed to connect to AWS')

    def _get_port(self):
        """Get the port. Where this information is retrieved from is engine dependent."""
        if self.data['Engine'] == 'memcached':
            return self.data['ConfigurationEndpoint']['Port']
        elif self.data['Engine'] == 'redis':
            return self.data['CacheNodes'][0]['Endpoint']['Port']

    def _refresh_data(self, cache_cluster_data=None):
        """Refresh data about this cache cluster"""
        if cache_cluster_data is None:
            try:
                response = self.conn.describe_cache_clusters(CacheClusterId=self.name, ShowCacheNodeInfo=True)
            except is_boto3_error_code('CacheClusterNotFound'):
                self.data = None
                self.status = 'gone'
                return
            except botocore.exceptions.ClientError as e:
                self.module.fail_json_aws(e, msg='Failed to describe cache clusters')
            cache_cluster_data = response['CacheClusters'][0]
        self.data = cache_cluster_data
        self.status = self.data['CacheClusterStatus']
        if self.status == 'rebooting cache cluster nodes':
            self.status = 'rebooting'

    def _get_nodes_to_remove(self):
        """If there are nodes to remove, it figures out which need to be removed"""
        num_nodes_to_remove = self.data['NumCacheNodes'] - self.num_nodes
        if num_nodes_to_remove <= 0:
            return []
        if not self.hard_modify:
            self.module.fail_json(msg=f"'{self.name}' requires removal of cache nodes. 'hard_modify' must be set to true to proceed.")
        cache_node_ids = [cn['CacheNodeId'] for cn in self.data['CacheNodes']]
        return cache_node_ids[-num_nodes_to_remove:]