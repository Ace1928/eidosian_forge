import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _LazyLoadOptions(self):
    """Lazily initializes descriptor options towards the end of the build."""
    if self._loaded_options:
        return
    from google.protobuf import descriptor_pb2
    if not hasattr(descriptor_pb2, self._options_class_name):
        raise RuntimeError('Unknown options class name %s!' % self._options_class_name)
    options_class = getattr(descriptor_pb2, self._options_class_name)
    features = None
    edition = self.file._edition
    if not self.has_options:
        if not self._features:
            features = self._ResolveFeatures(descriptor_pb2.Edition.Value(edition), options_class())
        with _lock:
            self._loaded_options = options_class()
            if not self._features:
                self._features = features
    else:
        if not self._serialized_options:
            options = self._options
        else:
            options = _ParseOptions(options_class(), self._serialized_options)
        if not self._features:
            features = self._ResolveFeatures(descriptor_pb2.Edition.Value(edition), options)
        with _lock:
            self._loaded_options = options
            if not self._features:
                self._features = features
            if options.HasField('features'):
                options.ClearField('features')
                if not options.SerializeToString():
                    self._loaded_options = options_class()
                    self.has_options = False