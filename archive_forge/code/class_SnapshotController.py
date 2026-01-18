from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class SnapshotController(object):

    def __init__(self, client, cluster_name):
        self.client = client
        self.cluster_name = cluster_name

    def get_cluster_snapshot_copy_status(self):
        response = self.client.describe_clusters(ClusterIdentifier=self.cluster_name)
        return response['Clusters'][0].get('ClusterSnapshotCopyStatus')

    def enable_snapshot_copy(self, destination_region, grant_name, retention_period):
        if grant_name:
            self.client.enable_snapshot_copy(ClusterIdentifier=self.cluster_name, DestinationRegion=destination_region, RetentionPeriod=retention_period, SnapshotCopyGrantName=grant_name)
        else:
            self.client.enable_snapshot_copy(ClusterIdentifier=self.cluster_name, DestinationRegion=destination_region, RetentionPeriod=retention_period)

    def disable_snapshot_copy(self):
        self.client.disable_snapshot_copy(ClusterIdentifier=self.cluster_name)

    def modify_snapshot_copy_retention_period(self, retention_period):
        self.client.modify_snapshot_copy_retention_period(ClusterIdentifier=self.cluster_name, RetentionPeriod=retention_period)