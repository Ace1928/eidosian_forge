from cloudsdk.google.protobuf import wrappers_pb2
class Int32ValueRule(WrapperRule):
    _proto_type = wrappers_pb2.Int32Value
    _python_type = int