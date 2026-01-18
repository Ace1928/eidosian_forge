from libcloud.common.types import (
class KeyPairError(LibcloudError):
    error_type = 'KeyPairError'

    def __init__(self, name, driver):
        self.name = name
        self.value = 'Key pair with name %s does not exist' % name
        super().__init__(value=self.value, driver=driver)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<{} name={}, value={}, driver={}>'.format(self.error_type, self.name, self.value, self.driver.name)