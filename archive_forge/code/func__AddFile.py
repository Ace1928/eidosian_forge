from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
def _AddFile(file_proto):
    for dependency in file_proto.dependency:
        if dependency in file_by_name:
            _AddFile(file_by_name.pop(dependency))
    _FACTORY.pool.Add(file_proto)