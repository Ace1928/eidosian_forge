from keystoneauth1 import adapter
from keystoneauth1 import session
from keystoneclient import client as ks_client
import manilaclient
from manilaclient.common import constants
from manilaclient.common import httpclient
from manilaclient import exceptions
from manilaclient.v2 import availability_zones
from manilaclient.v2 import limits
from manilaclient.v2 import messages
from manilaclient.v2 import quota_classes
from manilaclient.v2 import quotas
from manilaclient.v2 import resource_locks
from manilaclient.v2 import scheduler_stats
from manilaclient.v2 import security_services
from manilaclient.v2 import services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_backups
from manilaclient.v2 import share_export_locations
from manilaclient.v2 import share_group_snapshots
from manilaclient.v2 import share_group_type_access
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instance_export_locations
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_replica_export_locations
from manilaclient.v2 import share_replicas
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshot_export_locations
from manilaclient.v2 import share_snapshot_instance_export_locations
from manilaclient.v2 import share_snapshot_instances
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_transfers
from manilaclient.v2 import share_type_access
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
def _get_keystone_client(self):
    if self.insecure:
        verify = False
    else:
        verify = self.cacert or True
    ks_session = session.Session(verify=verify, cert=self.cert)
    ks_discover = session.discover.Discover(ks_session, self.auth_url)
    v2_auth_url = ks_discover.url_for('v2.0')
    v3_auth_url = ks_discover.url_for('v3.0')
    if v3_auth_url:
        keystone_client = ks_client.Client(session=ks_session, version=(3, 0), auth_url=v3_auth_url, username=self.username, password=self.password, user_id=self.user_id, user_domain_name=self.user_domain_name, user_domain_id=self.user_domain_id, project_id=self.project_id or self.tenant_id, project_name=self.project_name, project_domain_name=self.project_domain_name, project_domain_id=self.project_domain_id, region_name=self.region_name)
    elif v2_auth_url:
        keystone_client = ks_client.Client(session=ks_session, version=(2, 0), auth_url=v2_auth_url, username=self.username, password=self.password, tenant_id=self.tenant_id, tenant_name=self.tenant_name, region_name=self.region_name, cert=self.cert, use_keyring=self.use_keyring, force_new_token=self.force_new_token, stale_duration=self.cached_token_lifetime)
    else:
        raise exceptions.CommandError('Unable to determine the Keystone version to authenticate with using the given auth_url.')
    keystone_client.authenticate()
    return keystone_client