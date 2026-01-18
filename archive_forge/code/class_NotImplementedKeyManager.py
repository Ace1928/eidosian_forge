from castellan.key_manager import key_manager
class NotImplementedKeyManager(key_manager.KeyManager):
    """Key Manager Interface that raises NotImplementedError for all operations

    """

    def __init__(self, configuration=None):
        super(NotImplementedKeyManager, self).__init__(configuration)

    def create_key(self, context, algorithm='AES', length=256, expiration=None, name=None, **kwargs):
        raise NotImplementedError()

    def create_key_pair(self, context, algorithm, length, expiration=None, name=None):
        raise NotImplementedError()

    def store(self, context, managed_object, expiration=None, **kwargs):
        raise NotImplementedError()

    def copy(self, context, managed_object_id, **kwargs):
        raise NotImplementedError()

    def get(self, context, managed_object_id, **kwargs):
        raise NotImplementedError()

    def list(self, context, object_type=None):
        raise NotImplementedError()

    def delete(self, context, managed_object_id, force=False):
        raise NotImplementedError()

    def add_consumer(self, context, managed_object_id, consumer_data):
        raise NotImplementedError()

    def remove_consumer(self, context, managed_object_id, consumer_data):
        raise NotImplementedError()