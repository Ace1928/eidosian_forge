from lazr.restfulclient.resource import (
class CookbookWebServiceClient(ServiceRoot):
    RESOURCE_TYPE_CLASSES = dict(ServiceRoot.RESOURCE_TYPE_CLASSES)
    RESOURCE_TYPE_CLASSES['recipes'] = RecipeSet
    RESOURCE_TYPE_CLASSES['cookbooks'] = CookbookSet
    DEFAULT_SERVICE_ROOT = 'http://cookbooks.dev/'
    DEFAULT_VERSION = '1.0'

    def __init__(self, service_root=DEFAULT_SERVICE_ROOT, version=DEFAULT_VERSION, cache=None):
        super(CookbookWebServiceClient, self).__init__(None, service_root, cache=cache, version=version)