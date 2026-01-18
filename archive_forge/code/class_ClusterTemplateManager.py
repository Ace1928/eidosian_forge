from magnumclient.v1 import basemodels
class ClusterTemplateManager(basemodels.BaseModelManager):
    api_name = 'clustertemplates'
    resource_class = ClusterTemplate