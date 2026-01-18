from proto.primitives import ProtoType
class SFixed64Rule(StringyNumberRule):
    _python_type = int
    _proto_type = ProtoType.SFIXED64