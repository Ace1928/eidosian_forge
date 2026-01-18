import collections
import warnings
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import descriptor_database
from cloudsdk.google.protobuf import text_encoding
def NewFunc(*args, **kwargs):
    warnings.warn('Call to deprecated function %s(). Note: Do add unlinked descriptors to descriptor_pool is wrong. Use Add() or AddSerializedFile() instead.' % func.__name__, category=DeprecationWarning)
    return func(*args, **kwargs)