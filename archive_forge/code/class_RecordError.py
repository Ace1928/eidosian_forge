from libcloud.common.types import LibcloudError
class RecordError(LibcloudError):
    error_type = 'RecordError'

    def __init__(self, value, driver, record_id):
        self.record_id = record_id
        super().__init__(value=value, driver=driver)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<{} in {}, record_id={}, value={}>'.format(self.error_type, repr(self.driver), self.record_id, self.value)