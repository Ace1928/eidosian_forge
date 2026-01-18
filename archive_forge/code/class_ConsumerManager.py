from keystoneclient import base
from keystoneclient.v3.contrib.oauth1 import utils
class ConsumerManager(base.CrudManager):
    """Manager class for manipulating identity consumers."""
    resource_class = Consumer
    collection_key = 'consumers'
    key = 'consumer'
    base_url = utils.OAUTH_PATH

    def create(self, description=None, **kwargs):
        return super(ConsumerManager, self).create(description=description, **kwargs)

    def get(self, consumer):
        return super(ConsumerManager, self).get(consumer_id=base.getid(consumer))

    def update(self, consumer, description=None, **kwargs):
        return super(ConsumerManager, self).update(consumer_id=base.getid(consumer), description=description, **kwargs)

    def delete(self, consumer):
        return super(ConsumerManager, self).delete(consumer_id=base.getid(consumer))