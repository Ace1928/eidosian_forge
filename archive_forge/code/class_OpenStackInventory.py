import functools
from openstack.cloud import _utils
from openstack.config import loader
from openstack import connection
from openstack import exceptions
class OpenStackInventory:
    extra_config = None

    def __init__(self, config_files=None, refresh=False, private=False, config_key=None, config_defaults=None, cloud=None, use_direct_get=False):
        if config_files is None:
            config_files = []
        config = loader.OpenStackConfig(config_files=loader.CONFIG_FILES + config_files)
        self.extra_config = config.get_extra_config(config_key, config_defaults)
        if cloud is None:
            self.clouds = [connection.Connection(config=cloud_region) for cloud_region in config.get_all()]
        else:
            self.clouds = [connection.Connection(config=config.get_one(cloud))]
        if private:
            for cloud in self.clouds:
                cloud.private = True
        if refresh:
            for cloud in self.clouds:
                cloud._cache.invalidate()

    def list_hosts(self, expand=True, fail_on_cloud_config=True, all_projects=False):
        hostvars = []
        for cloud in self.clouds:
            try:
                for server in cloud.list_servers(detailed=expand, all_projects=all_projects):
                    hostvars.append(server)
            except exceptions.SDKException:
                if fail_on_cloud_config:
                    raise
        return hostvars

    def search_hosts(self, name_or_id=None, filters=None, expand=True):
        hosts = self.list_hosts(expand=expand)
        return _utils._filter_list(hosts, name_or_id, filters)

    def get_host(self, name_or_id, filters=None, expand=True):
        if expand:
            func = self.search_hosts
        else:
            func = functools.partial(self.search_hosts, expand=False)
        return _utils._get_entity(self, func, name_or_id, filters)