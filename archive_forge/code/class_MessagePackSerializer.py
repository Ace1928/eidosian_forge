from oslo_serialization import msgpackutils
from oslo_serialization.serializer.base_serializer import BaseSerializer
class MessagePackSerializer(BaseSerializer):
    """MessagePack serializer based on the msgpackutils module."""

    def __init__(self, registry=None):
        self._registry = registry

    def dump(self, obj, fp):
        return msgpackutils.dump(obj, fp, registry=self._registry)

    def dump_as_bytes(self, obj):
        return msgpackutils.dumps(obj, registry=self._registry)

    def load(self, fp):
        return msgpackutils.load(fp, registry=self._registry)

    def load_from_bytes(self, s):
        return msgpackutils.loads(s, registry=self._registry)