import abc
from oslo_serialization import jsonutils
class NoOpSerializer(Serializer):
    """A serializer that does nothing."""

    def serialize_entity(self, ctxt, entity):
        return entity

    def deserialize_entity(self, ctxt, entity):
        return entity

    def serialize_context(self, ctxt):
        return ctxt

    def deserialize_context(self, ctxt):
        return ctxt