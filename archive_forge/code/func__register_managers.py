from keystoneauth1 import adapter
from saharaclient.api import cluster_templates
from saharaclient.api import clusters
from saharaclient.api import data_sources
from saharaclient.api import images
from saharaclient.api import job_binaries
from saharaclient.api import job_binary_internals
from saharaclient.api import job_executions
from saharaclient.api import job_types
from saharaclient.api import jobs
from saharaclient.api import node_group_templates
from saharaclient.api import plugins
from saharaclient.api.v2 import job_templates
from saharaclient.api.v2 import jobs as jobs_v2
def _register_managers(self, client):
    self.clusters = clusters.ClusterManagerV2(client)
    self.cluster_templates = cluster_templates.ClusterTemplateManagerV2(client)
    self.node_group_templates = node_group_templates.NodeGroupTemplateManagerV2(client)
    self.plugins = plugins.PluginManagerV2(client)
    self.images = images.ImageManagerV2(client)
    self.data_sources = data_sources.DataSourceManagerV2(client)
    self.job_templates = job_templates.JobTemplatesManagerV2(client)
    self.jobs = jobs_v2.JobsManagerV2(client)
    self.job_binaries = job_binaries.JobBinariesManagerV2(client)
    self.job_types = job_types.JobTypesManager(client)