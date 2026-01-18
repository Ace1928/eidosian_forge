import abc
from oslo_serialization import jsonutils
class JsonPayloadSerializer(NoOpSerializer):

    @staticmethod
    def serialize_entity(context, entity):
        return jsonutils.to_primitive(entity, convert_instances=True)