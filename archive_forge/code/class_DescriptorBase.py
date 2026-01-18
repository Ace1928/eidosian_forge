import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class DescriptorBase(metaclass=DescriptorMetaclass):
    """Descriptors base class.

  This class is the base of all descriptor classes. It provides common options
  related functionality.

  Attributes:
    has_options:  True if the descriptor has non-default options.  Usually it is
      not necessary to read this -- just call GetOptions() which will happily
      return the default instance.  However, it's sometimes useful for
      efficiency, and also useful inside the protobuf implementation to avoid
      some bootstrapping issues.
    file (FileDescriptor): Reference to file info.
  """
    if _USE_C_DESCRIPTORS:
        _C_DESCRIPTOR_CLASS = ()

    def __init__(self, file, options, serialized_options, options_class_name):
        """Initialize the descriptor given its options message and the name of the
    class of the options message. The name of the class is required in case
    the options message is None and has to be created.
    """
        self._features = None
        self.file = file
        self._options = options
        self._loaded_options = None
        self._options_class_name = options_class_name
        self._serialized_options = serialized_options
        self.has_options = self._options is not None or self._serialized_options is not None

    @property
    @abc.abstractmethod
    def _parent(self):
        pass

    def _InferLegacyFeatures(self, edition, options, features):
        """Infers features from proto2/proto3 syntax so that editions logic can be used everywhere.

    Args:
      edition: The edition to infer features for.
      options: The options for this descriptor that are being processed.
      features: The feature set object to modify with inferred features.
    """
        pass

    def _GetFeatures(self):
        if not self._features:
            self._LazyLoadOptions()
        return self._features

    def _ResolveFeatures(self, edition, raw_options):
        """Resolves features from the raw options of this descriptor.

    Args:
      edition: The edition to use for feature defaults.
      raw_options: The options for this descriptor that are being processed.

    Returns:
      A fully resolved feature set for making runtime decisions.
    """
        from google.protobuf import descriptor_pb2
        if self._parent:
            features = descriptor_pb2.FeatureSet()
            features.CopyFrom(self._parent._GetFeatures())
        else:
            features = self.file.pool._CreateDefaultFeatures(edition)
        unresolved = descriptor_pb2.FeatureSet()
        unresolved.CopyFrom(raw_options.features)
        self._InferLegacyFeatures(edition, raw_options, unresolved)
        features.MergeFrom(unresolved)
        return self.file.pool._InternFeatures(features)

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

    def GetOptions(self):
        """Retrieves descriptor options.

    Returns:
      The options set on this descriptor.
    """
        if not self._loaded_options:
            self._LazyLoadOptions()
        return self._loaded_options