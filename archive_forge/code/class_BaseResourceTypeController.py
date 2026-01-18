import testtools
class BaseResourceTypeController(BaseController):

    def __init__(self, api, schema_api, controller_class):
        super(BaseResourceTypeController, self).__init__(api, schema_api, controller_class)

    def get(self, *args, **kwargs):
        resource_types = self.controller.get(*args)
        names = [rt.name for rt in resource_types]
        self._assertRequestId(resource_types)
        return names