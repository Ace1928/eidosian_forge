from oslo_log import log
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.auth import core
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.identity.backends import resource_options as ro
class _ReceiptModelHandler(object):
    identity = 125
    handles = (ReceiptModel,)

    def __init__(self, registry):
        self._registry = registry

    def serialize(self, obj):
        serialized = msgpackutils.dumps(obj.__dict__, registry=self._registry)
        return serialized

    def deserialize(self, data):
        receipt_data = msgpackutils.loads(data, registry=self._registry)
        try:
            receipt_model = ReceiptModel()
            for k, v in iter(receipt_data.items()):
                setattr(receipt_model, k, v)
        except Exception:
            LOG.debug('Failed to deserialize ReceiptModel. Data is %s', receipt_data)
            raise exception.CacheDeserializationError(ReceiptModel.__name__, receipt_data)
        return receipt_model