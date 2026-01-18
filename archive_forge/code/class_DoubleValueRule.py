from cloudsdk.google.protobuf import wrappers_pb2
class DoubleValueRule(WrapperRule):
    _proto_type = wrappers_pb2.DoubleValue
    _python_type = float