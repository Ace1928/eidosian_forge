import abc
from tensorflow.python.util.tf_export import tf_export
Creates a FeatureColumn from its config.

    This method should be the reverse of `get_config`, capable of instantiating
    the same FeatureColumn from the config dictionary. See `get_config` for an
    example of common (de)serialization practices followed in this file.

    TODO(b/118939620): This is a private method until consensus is reached on
    supporting object deserialization deduping within Keras.

    Args:
      config: A Dict config acquired with `get_config`.
      custom_objects: Optional dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization.
      columns_by_name: A Dict[String, FeatureColumn] of existing columns in
        order to avoid duplication. Should be passed to any calls to
        deserialize_feature_column().

    Returns:
      A FeatureColumn for the input config.
    