from cloudsdk.google.protobuf import wrappers_pb2
class BoolValueRule(WrapperRule):
    _proto_type = wrappers_pb2.BoolValue
    _python_type = bool