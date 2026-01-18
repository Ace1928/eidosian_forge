import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
def _merge_features(self, preprocessed_features, crossed_features):
    if not self._preprocessed_features_names:
        self._preprocessed_features_names = sorted(preprocessed_features.keys())
        self._crossed_features_names = sorted(crossed_features.keys())
    all_names = self._preprocessed_features_names + self._crossed_features_names
    all_features = [preprocessed_features[name] for name in self._preprocessed_features_names] + [crossed_features[name] for name in self._crossed_features_names]
    if self.output_mode == 'dict':
        output_dict = {}
    else:
        features_to_concat = []
    if self._sublayers_built:
        for name, feature in zip(all_names, all_features):
            encoder = self.one_hot_encoders.get(name, None)
            if encoder:
                feature = encoder(feature)
            if self.output_mode == 'dict':
                output_dict[name] = feature
            else:
                features_to_concat.append(feature)
        if self.output_mode == 'dict':
            return output_dict
        else:
            return self.concat(features_to_concat)
    all_specs = [self.features[name] for name in self._preprocessed_features_names] + [self.crosses_by_name[name] for name in self._crossed_features_names]
    for name, feature, spec in zip(all_names, all_features, all_specs):
        if tree.is_nested(feature):
            dtype = tree.flatten(feature)[0].dtype
        else:
            dtype = feature.dtype
        dtype = backend.standardize_dtype(dtype)
        if spec.output_mode == 'one_hot':
            preprocessor = self.preprocessors.get(name) or self.crossers.get(name)
            cardinality = None
            if not dtype.startswith('int'):
                raise ValueError(f"Feature '{name}' has `output_mode='one_hot'`. Thus its preprocessor should return an integer dtype. Instead it returns a {dtype} dtype.")
            if isinstance(preprocessor, (layers.IntegerLookup, layers.StringLookup)):
                cardinality = preprocessor.vocabulary_size()
            elif isinstance(preprocessor, layers.CategoryEncoding):
                cardinality = preprocessor.num_tokens
            elif isinstance(preprocessor, layers.Discretization):
                cardinality = preprocessor.num_bins
            elif isinstance(preprocessor, (layers.HashedCrossing, layers.Hashing)):
                cardinality = preprocessor.num_bins
            else:
                raise ValueError(f"Feature '{name}' has `output_mode='one_hot'`. However it isn't a standard feature and the dimensionality of its output space is not known, thus it cannot be one-hot encoded. Try using `output_mode='int'`.")
            if cardinality is not None:
                encoder = layers.CategoryEncoding(num_tokens=cardinality, output_mode='multi_hot')
                self.one_hot_encoders[name] = encoder
                feature = encoder(feature)
        if self.output_mode == 'concat':
            dtype = feature.dtype
            if dtype.startswith('int') or dtype == 'string':
                raise ValueError(f"Cannot concatenate features because feature '{name}' has not been encoded (it has dtype {dtype}). Consider using `output_mode='dict'`.")
            features_to_concat.append(feature)
        else:
            output_dict[name] = feature
    if self.output_mode == 'concat':
        self.concat = layers.Concatenate(axis=-1)
        return self.concat(features_to_concat)
    else:
        return output_dict