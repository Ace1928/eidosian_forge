from urllib import parse as urlparse
from saharaclient.api import base
class PluginManagerV1(_PluginManager):

    def convert_to_cluster_template(self, plugin_name, hadoop_version, template_name, filecontent):
        """Convert to cluster template

        Create Cluster Template directly, avoiding Cluster Template
        mechanism.
        """
        resp = self.api.post('/plugins/%s/%s/convert-config/%s' % (plugin_name, hadoop_version, urlparse.quote(template_name)), data=filecontent)
        if resp.status_code != 202:
            raise RuntimeError('Failed to upload template file for plugin "%s" and version "%s"' % (plugin_name, hadoop_version))
        else:
            return base.get_json(resp)['cluster_template']