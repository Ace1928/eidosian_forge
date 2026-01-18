from libcloud.common.types import LibcloudError
class ObjectError(LibcloudError):
    error_type = 'ContainerError'

    def __init__(self, value, driver, object_name):
        self.object_name = object_name
        super().__init__(value=value, driver=driver)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<{} in {}, value={}, object = {}>'.format(self.error_type, repr(self.driver), self.value, self.object_name)