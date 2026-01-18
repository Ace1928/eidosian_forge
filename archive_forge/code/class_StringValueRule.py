from cloudsdk.google.protobuf import wrappers_pb2
class StringValueRule(WrapperRule):
    _proto_type = wrappers_pb2.StringValue
    _python_type = str