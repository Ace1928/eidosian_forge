import collections
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
class IndexLookup(base_preprocessing_layer.PreprocessingLayer):
    """Maps values from a vocabulary to integer indices.

    This layer translates a set of arbitrary hashables into an integer output
    via a table-based lookup, with optional out-of-vocabulary handling. This is
    the basis layer for both IntegerLookup and StringLookup; it holds the common
    logic but is not intended to be exported as part of the Keras API.

    Args:
      max_tokens: The maximum size of the vocabulary for this layer. If None,
        there is no cap on the size of the vocabulary. Note that this size
        includes the OOV and mask tokens.
      num_oov_indices: The number of out-of-vocabulary tokens to use. If this
        value is more than 1, OOV inputs are hashed to determine their OOV
        value. If this value is 0, OOV inputs will cause an error when calling
        the layer.
      mask_token: A token that represents masked inputs. When `output_mode` is
        `"int"`, the token is included in vocabulary and mapped to index 0. In
        other output modes, the token will not appear in the vocabulary and
        instances of the mask token in the input will be dropped. If set to
        None, no mask term will be added.
      oov_token: Only used when `invert` is True. The token to return for OOV
        indices.
      vocabulary: Optional. Either an array or a string path to a text file. If
        passing an array, can pass a tuple, list, 1D numpy array, or 1D tensor
        containing the vocbulary terms. If passing a file path, the file should
        contain one line per term in the vocabulary. If this argument is set,
        there is no need to `adapt` the layer.
      vocabulary_dtype: The dtype of the vocabulary terms. For example,
        `"int64"` or `"string"`.
      idf_weights: Only valid when `output_mode` is `"tf_idf"`. A tuple, list,
        1D numpy array, or 1D tensor or the same length as the vocabulary,
        containing the floating point inverse document frequency weights, which
        will be multiplied by per sample term counts for the final `tf_idf`
        weight. If the `vocabulary` argument is set, and `output_mode` is
        `"tf_idf"`, this argument must be supplied.
      invert: Only valid when `output_mode` is `"int"`. If True, this layer will
        map indices to vocabulary items instead of mapping vocabulary items to
        indices. Defaults to `False`.
      output_mode: Specification for the output of the layer. Values can be
        `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"`
        configuring the layer as follows:
          - `"int"`: Return the raw integer indices of the input tokens.
          - `"one_hot"`: Encodes each individual element in the input into an
            array the same size as the vocabulary, containing a 1 at the element
            index. If the last dimension is size 1, will encode on that
            dimension.  If the last dimension is not size 1, will append a new
            dimension for the encoded output.
          - `"multi_hot"`: Encodes each sample in the input into a single array
            the same size as the vocabulary, containing a 1 for each vocabulary
            term present in the sample. Treats the last dimension as the sample
            dimension, if input shape is (..., sample_length), output shape will
            be (..., num_tokens).
          - `"count"`: As `"multi_hot"`, but the int array contains a count of
            the number of times the token at that index appeared in the sample.
          - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
            find the value in each token slot.
        Defaults to `"int"`.
      pad_to_max_tokens: Only valid when `output_mode` is `"multi_hot"`,
        `"count"`, or `"tf_idf"`. If True, the output will have its feature axis
        padded to `max_tokens` even if the number of unique tokens in the
        vocabulary is less than max_tokens, resulting in a tensor of shape
        [batch_size, max_tokens] regardless of vocabulary size. Defaults to
        False.
      sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`, `"count"`
        and `"tf-idf"` output modes. If True, returns a `SparseTensor` instead
        of a dense `Tensor`. Defaults to `False`.
    """

    def __init__(self, max_tokens, num_oov_indices, mask_token, oov_token, vocabulary_dtype, vocabulary=None, idf_weights=None, invert=False, output_mode='int', sparse=False, pad_to_max_tokens=False, **kwargs):
        if max_tokens is not None and max_tokens <= 1:
            raise ValueError(f'If set, `max_tokens` must be greater than 1. Received: max_tokens={max_tokens}')
        if pad_to_max_tokens and max_tokens is None:
            raise ValueError(f'If pad_to_max_tokens is True, must set `max_tokens`. Received: max_tokens={max_tokens}')
        if num_oov_indices < 0:
            raise ValueError(f'`num_oov_indices` must be greater than or equal to 0. Received: num_oov_indices={num_oov_indices}')
        if output_mode == 'binary':
            output_mode = MULTI_HOT
        if output_mode == 'tf-idf':
            output_mode = TF_IDF
        layer_utils.validate_string_arg(output_mode, allowable_strings=(INT, ONE_HOT, MULTI_HOT, COUNT, TF_IDF), layer_name=self.__class__.__name__, arg_name='output_mode')
        if invert and output_mode != INT:
            raise ValueError(f"`output_mode` must be `'int'` when `invert` is true. Received: output_mode={output_mode}")
        if sparse and output_mode == INT:
            raise ValueError(f"`sparse` may only be true if `output_mode` is `'one_hot'`, `'multi_hot'`, `'count'` or `'tf_idf'`. Received: sparse={sparse} and output_mode={output_mode}")
        if idf_weights is not None and output_mode != TF_IDF:
            raise ValueError(f"`idf_weights` should only be set if `output_mode` is `'tf_idf'`. Received: idf_weights={idf_weights} and output_mode={output_mode}")
        self.invert = invert
        self.max_tokens = max_tokens
        self.num_oov_indices = num_oov_indices
        self.mask_token = mask_token
        self.oov_token = oov_token
        self.output_mode = output_mode
        self.sparse = sparse
        self.pad_to_max_tokens = pad_to_max_tokens
        self.vocabulary_dtype = vocabulary_dtype
        self._frozen_vocab_size = kwargs.pop('vocabulary_size', None)
        self.input_vocabulary = vocabulary
        self.input_idf_weights = idf_weights
        self._has_input_vocabulary = kwargs.pop('has_input_vocabulary', vocabulary is not None)
        kwargs.pop('has_static_table', None)
        if 'dtype' not in kwargs:
            kwargs['dtype'] = tf.int64 if output_mode == INT else backend.floatx()
        super().__init__(**kwargs)
        if output_mode == INT and (not tf.as_dtype(self.compute_dtype).is_integer):
            input_dtype = kwargs['dtype']
            raise ValueError(f"When `output_mode='int'`, `dtype` should be an integer type. Received: dtype={input_dtype}")
        if invert:
            self._key_dtype = self.dtype if output_mode == INT else tf.int64
            self._value_dtype = tf.as_dtype(self.vocabulary_dtype)
            mask_key = 0
            mask_value = mask_token
            self._default_value = self.oov_token
        else:
            self._key_dtype = tf.as_dtype(self.vocabulary_dtype)
            self._value_dtype = self.dtype if output_mode == INT else tf.int64
            mask_key = mask_token
            mask_value = 0 if self.output_mode == INT else self._value_dtype.max
            if self.num_oov_indices == 0:
                self._default_value = -1
            elif self.num_oov_indices == 1:
                self._default_value = self._oov_start_index()
            else:
                self._default_value = -1
        if self.mask_token is not None:
            self._mask_key = tf.convert_to_tensor(mask_key, self._key_dtype)
            self._mask_value = tf.convert_to_tensor(mask_value, self._value_dtype)
        if self.output_mode == TF_IDF:
            self.idf_weights = tf.Variable([0] * self._token_start_index(), shape=(None,), dtype=self.compute_dtype, trainable=False)
            self.idf_weights_const = self.idf_weights.value()
        if vocabulary is not None:
            self.set_vocabulary(vocabulary, idf_weights)
        else:
            self.lookup_table = self._uninitialized_lookup_table()
        if not self._has_input_vocabulary:
            self._add_trackable(VocabWeightHandler(self), False)
            self.token_counts = tf.lookup.experimental.MutableHashTable(key_dtype=vocabulary_dtype, value_dtype=tf.int64, default_value=0)
            if self.output_mode == TF_IDF:
                self.token_document_counts = tf.lookup.experimental.MutableHashTable(key_dtype=vocabulary_dtype, value_dtype=tf.int64, default_value=0)
                self.num_documents = tf.Variable(0, dtype=tf.int64, trainable=False)

    def compute_output_shape(self, input_shape):
        if self.output_mode == INT:
            return input_shape
        depth = self.max_tokens if self.pad_to_max_tokens else self._frozen_vocab_size
        return tf.TensorShape([input_shape[0], depth])

    def compute_output_signature(self, input_spec):
        output_shape = self.compute_output_shape(input_spec.shape.as_list())
        output_dtype = self.vocabulary_dtype if self.invert else self.compute_dtype
        return tf.TensorSpec(shape=output_shape, dtype=output_dtype)

    def get_vocabulary(self, include_special_tokens=True):
        """Returns the current vocabulary of the layer.

        Args:
          include_special_tokens: If True, the returned vocabulary will include
            mask and OOV tokens, and a term's index in the vocabulary will equal
            the term's index when calling the layer. If False, the returned
            vocabulary will not include any mask or OOV tokens.
        """
        if self.lookup_table.size() == 0:
            vocab, indices = ([], [])
        else:
            keys, values = self.lookup_table.export()
            vocab, indices = (values, keys) if self.invert else (keys, values)
            vocab, indices = (self._tensor_vocab_to_numpy(vocab), indices.numpy())
        lookup = collections.defaultdict(lambda: self.oov_token, zip(indices, vocab))
        vocab = [lookup[x] for x in range(self.vocabulary_size())]
        if self.mask_token is not None and self.output_mode == INT:
            vocab[0] = self.mask_token
        if not include_special_tokens:
            vocab = vocab[self._token_start_index():]
        return vocab

    def vocabulary_size(self):
        """Gets the current size of the layer's vocabulary.

        Returns:
          The integer size of the vocabulary, including optional mask and oov
          indices.
        """
        if tf.executing_eagerly():
            return int(self.lookup_table.size().numpy()) + self._token_start_index()
        else:
            return self.lookup_table.size() + self._token_start_index()

    def vocab_size(self):
        logging.warning('vocab_size is deprecated, please use vocabulary_size.')
        return self.vocabulary_size()

    def get_config(self):
        config = {'invert': self.invert, 'max_tokens': self.max_tokens, 'num_oov_indices': self.num_oov_indices, 'oov_token': self.oov_token, 'mask_token': self.mask_token, 'output_mode': self.output_mode, 'sparse': self.sparse, 'pad_to_max_tokens': self.pad_to_max_tokens, 'vocabulary_dtype': self.vocabulary_dtype, 'idf_weights': utils.listify_tensors(self.input_idf_weights), 'vocabulary': utils.listify_tensors(self.input_vocabulary), 'vocabulary_size': self._frozen_vocab_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _record_vocabulary_size(self):
        self._ensure_vocab_size_unchanged()
        with tf.init_scope():
            self._frozen_vocab_size = self.vocabulary_size()

    def set_vocabulary(self, vocabulary, idf_weights=None):
        """Sets vocabulary (and optionally document frequency) for this layer.

        This method sets the vocabulary and idf weights for this layer directly,
        instead of analyzing a dataset through `adapt`. It should be used
        whenever the vocab (and optionally document frequency) information is
        already known.  If vocabulary data is already present in the layer, this
        method will replace it.

        Args:
          vocabulary: Either an array or a string path to a text file. If
            passing an array, can pass a tuple, list, 1D numpy array, or 1D
            tensor containing the vocbulary terms. If passing a file path, the
            file should contain one line per term in the vocabulary.
          idf_weights: A tuple, list, 1D numpy array, or 1D tensor of inverse
            document frequency weights with equal length to vocabulary. Must be
            set if `output_mode` is `"tf_idf"`. Should not be set otherwise.

        Raises:
          ValueError: If there are too many inputs, the inputs do not match, or
            input data is missing.
          RuntimeError: If the vocabulary cannot be set when this function is
            called. This happens when `"multi_hot"`, `"count"`, and `"tf_idf"`
            modes, if `pad_to_max_tokens` is False and the layer itself has
            already been called.
          RuntimeError: If a tensor vocabulary is passed outside of eager
            execution.
        """
        if self.output_mode == TF_IDF:
            if idf_weights is None:
                raise ValueError('`idf_weights` must be set if output_mode is TF_IDF')
        elif idf_weights is not None:
            raise ValueError(f"`idf_weights` should only be set if output_mode is `'tf_idf'`. Received: output_mode={self.output_mode} and idf_weights={idf_weights}")
        if isinstance(vocabulary, str):
            if not tf.io.gfile.exists(vocabulary):
                raise ValueError(f'Vocabulary file {vocabulary} does not exist.')
            if self.output_mode == TF_IDF:
                raise ValueError("output_mode `'tf_idf'` does not support loading a vocabulary from file.")
            self.lookup_table = self._lookup_table_from_file(vocabulary)
            self._record_vocabulary_size()
            return
        if not tf.executing_eagerly() and (tf.is_tensor(vocabulary) or tf.is_tensor(idf_weights)):
            raise RuntimeError('Cannot set a tensor vocabulary on {} layer {} when not executing eagerly. Create this layer or call `set_vocabulary` outside of any `tf.function`s and with eager execution enabled.'.format(self.__class__.__name__, self.name))
        if tf.is_tensor(vocabulary):
            vocabulary = self._tensor_vocab_to_numpy(vocabulary)
        elif isinstance(vocabulary, (list, tuple)):
            vocabulary = np.array(vocabulary)
        if tf.is_tensor(idf_weights):
            idf_weights = idf_weights.numpy()
        elif isinstance(idf_weights, (list, tuple)):
            idf_weights = np.array(idf_weights)
        if vocabulary.size == 0:
            raise ValueError(f'Cannot set an empty vocabulary, you passed {vocabulary}.')
        oov_start = self._oov_start_index()
        token_start = self._token_start_index()
        special_tokens = [self.mask_token] * oov_start + [self.oov_token] * self.num_oov_indices
        found_special_tokens = np.array_equal(special_tokens, vocabulary[:token_start])
        if found_special_tokens:
            tokens = vocabulary[token_start:]
        else:
            tokens = vocabulary
        repeated_tokens = self._find_repeated_tokens(tokens)
        if repeated_tokens:
            raise ValueError('The passed vocabulary has at least one repeated term. Please uniquify your dataset. The repeated terms are {}'.format(repeated_tokens))
        if self.mask_token is not None and self.mask_token in tokens:
            mask_index = np.argwhere(vocabulary == self.mask_token)[-1]
            raise ValueError(f'Found reserved mask token at unexpected location in `vocabulary`. Note that passed `vocabulary` does not need to include the OOV and mask tokens. Either remove all mask and OOV tokens, or include them only at the start of the vocabulary in precisely this order: {special_tokens}. Received: mask_token={self.mask_token} at vocabulary index {mask_index}')
        if self.oov_token is not None and self.invert and (self.oov_token in tokens):
            oov_index = np.argwhere(vocabulary == self.oov_token)[-1]
            raise ValueError(f'Found reserved OOV token at unexpected location in `vocabulary`. Note that passed `vocabulary` does not need to include the OOV and mask tokens. Either remove all mask and OOV tokens, or include them only at the start of the vocabulary in precisely this order: {special_tokens}. Received: oov_token={self.oov_token} at vocabulary index {oov_index}')
        new_vocab_size = token_start + len(tokens)
        if self.max_tokens is not None and new_vocab_size > self.max_tokens:
            raise ValueError('Attempted to set a vocabulary larger than the maximum vocab size. Passed vocab size is {}, max vocab size is {}.'.format(new_vocab_size, self.max_tokens))
        self.lookup_table = self._lookup_table_from_tokens(tokens)
        self._record_vocabulary_size()
        if self.output_mode == TF_IDF and idf_weights is not False:
            if len(vocabulary) != len(idf_weights):
                raise ValueError('`idf_weights` must be the same length as vocabulary. len(idf_weights) is {}, len(vocabulary) is {}'.format(len(vocabulary), len(idf_weights)))
            idf_weights = self._convert_to_ndarray(idf_weights)
            if idf_weights.ndim != 1:
                raise ValueError('TF-IDF data must be a 1-index array, but received {}'.format(type(idf_weights)))
            if found_special_tokens:
                front_padding = 0
                front_padding_value = 0
            else:
                front_padding = token_start
                front_padding_value = np.average(idf_weights)
            back_padding_value = 0
            if self.pad_to_max_tokens and self.max_tokens is not None:
                back_padding = self.max_tokens - front_padding - len(idf_weights)
            else:
                back_padding = 0
            weights = np.pad(idf_weights, (front_padding, back_padding), 'constant', constant_values=(front_padding_value, back_padding_value))
            weights = tf.convert_to_tensor(weights, dtype=self.compute_dtype)
            self.idf_weights.assign(weights)
            self.idf_weights_const = self.idf_weights.value()

    def update_state(self, data):
        if self._has_input_vocabulary:
            raise ValueError('Cannot adapt {} layer after setting a static vocabulary via init argument or `set_vocabulary`.'.format(self.__class__.__name__))
        data = utils.ensure_tensor(data, dtype=self.vocabulary_dtype)
        if data.shape.rank == 0:
            data = tf.expand_dims(data, 0)
        if data.shape.rank == 1:
            data = tf.expand_dims(data, 0)
        tokens, counts = self._num_tokens(data)
        self.token_counts.insert(tokens, counts + self.token_counts.lookup(tokens))
        if self.output_mode == TF_IDF:
            deduped_doc_data = tf.map_fn(lambda x: tf.unique(x)[0], data)
            tokens, doc_counts = self._num_tokens(deduped_doc_data)
            self.token_document_counts.insert(tokens, doc_counts + self.token_document_counts.lookup(tokens))
            if tf_utils.is_ragged(data):
                self.num_documents.assign_add(data.nrows())
            else:
                self.num_documents.assign_add(tf.shape(data, out_type=tf.int64)[0])

    def finalize_state(self):
        if self._has_input_vocabulary or tf.equal(self.token_counts.size(), 0):
            if self.output_mode == TF_IDF:
                self.idf_weights_const = self.idf_weights.value()
            self._record_vocabulary_size()
            return
        if self.mask_token is not None:
            self.token_counts.remove(tf.convert_to_tensor([self.mask_token], self.vocabulary_dtype))
        if self.oov_token is not None:
            self.token_counts.remove(tf.convert_to_tensor([self.oov_token], self.vocabulary_dtype))
        tokens, counts = self.token_counts.export()
        sorted_indices = np.lexsort((tokens.numpy(), counts.numpy()))[::-1]
        token_start = self._token_start_index()
        if self.max_tokens:
            max_learned_tokens = self.max_tokens - token_start
            sorted_indices = sorted_indices[:max_learned_tokens]
        tokens = tf.gather(tokens, sorted_indices)
        self.lookup_table = self._lookup_table_from_tokens(tokens)
        if self.output_mode == TF_IDF:
            token_document_counts = self.token_document_counts.lookup(tokens)
            idf_weights = self._inverse_document_frequency(token_document_counts, self.num_documents)
            idf_weights = tf.cast(idf_weights, self.compute_dtype)
            idf_weights = tf.pad(idf_weights, [[self._token_start_index(), 0]], constant_values=tf.reduce_mean(idf_weights))
            if self.pad_to_max_tokens and self.max_tokens is not None:
                idf_weights = tf.pad(idf_weights, [[0, self.max_tokens - tf.size(idf_weights)]], constant_values=0)
            self.idf_weights.assign(idf_weights)
            self.idf_weights_const = self.idf_weights.value()
        self.reset_state()
        self._record_vocabulary_size()

    def reset_state(self):
        if self._has_input_vocabulary:
            return
        self.token_counts.remove(self.token_counts.export()[0])
        if self.output_mode == TF_IDF:
            self.token_document_counts.remove(self.token_document_counts.export()[0])
            self.num_documents.assign(0)

    def call(self, inputs):
        self._ensure_known_vocab_size()
        inputs = utils.ensure_tensor(inputs, dtype=self._key_dtype)
        original_shape = inputs.shape
        if inputs.shape.rank == 0:
            inputs = self._expand_dims(inputs, -1)
        if tf_utils.is_sparse(inputs):
            lookups = tf.SparseTensor(inputs.indices, self._lookup_dense(inputs.values), inputs.dense_shape)
        elif tf_utils.is_ragged(inputs):
            lookups = tf.ragged.map_flat_values(self._lookup_dense, inputs)
        else:
            lookups = self._lookup_dense(inputs)
        if self.output_mode == INT:
            if original_shape.rank == 0:
                lookups = tf.squeeze(lookups, -1)
            return lookups
        depth = self.max_tokens if self.pad_to_max_tokens else self._frozen_vocab_size
        idf_weights = self.idf_weights_const if self.output_mode == TF_IDF else None
        return utils.encode_categorical_inputs(lookups, output_mode=self.output_mode, depth=depth, dtype=self.compute_dtype, sparse=self.sparse, idf_weights=idf_weights)

    def _lookup_dense(self, inputs):
        """Lookup table values for a dense Tensor, handling masking and OOV."""
        if tf.executing_eagerly() and backend.is_keras_tensor(inputs):
            lookups = tf.zeros_like(inputs, dtype=self._value_dtype)
        else:
            lookups = self.lookup_table.lookup(inputs)
        if self.mask_token is not None:
            mask_locations = tf.equal(inputs, self._mask_key)
            lookups = tf.where(mask_locations, self._mask_value, lookups)
        if self.invert:
            return lookups
        lookup_checks = []
        if self.num_oov_indices == 0:
            oov_indices = tf.where(tf.equal(lookups, -1))
            oov_inputs = tf.gather_nd(inputs, oov_indices)
            msg = tf.strings.format('When `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.', (oov_inputs,))
            assertion = tf.Assert(tf.equal(tf.size(oov_indices), 0), [msg])
            lookup_checks.append(assertion)
        elif self.num_oov_indices > 1:
            if self._key_dtype.is_integer:
                oov_indices = tf.math.floormod(inputs, self.num_oov_indices)
            else:
                oov_indices = tf.strings.to_hash_bucket_fast(inputs, num_buckets=self.num_oov_indices)
            oov_indices = oov_indices + self._oov_start_index()
            oov_locations = tf.equal(lookups, self._default_value)
            lookups = tf.where(oov_locations, oov_indices, lookups)
        with tf.control_dependencies(lookup_checks):
            return tf.identity(lookups)

    def save_own_variables(self, store):
        if self.output_mode == TF_IDF:
            store['idf_weights'] = self.idf_weights_const.numpy()

    def load_own_variables(self, store):
        if self.output_mode == TF_IDF:
            self.idf_weights.assign(store['idf_weights'])
            self.idf_weights_const = self.idf_weights.value()

    def save_assets(self, dir_path):
        if self.input_vocabulary:
            return
        vocabulary = self.get_vocabulary(include_special_tokens=True)
        vocabulary_filepath = tf.io.gfile.join(dir_path, 'vocabulary.txt')
        with open(vocabulary_filepath, 'w') as f:
            f.write('\n'.join([str(w) for w in vocabulary]))

    def load_assets(self, dir_path):
        if self.input_vocabulary:
            return
        vocabulary_filepath = tf.io.gfile.join(dir_path, 'vocabulary.txt')
        with open(vocabulary_filepath, 'r') as f:
            lines = f.read().split('\n')
            if tf.as_dtype(self.vocabulary_dtype) == tf.string:
                values = [str(line) for line in lines]
            else:
                values = [int(line) for line in lines]
            if self.output_mode == TF_IDF:
                self.set_vocabulary(values, idf_weights=False)
            else:
                self.set_vocabulary(values)

    def _uninitialized_lookup_table(self):
        with tf.init_scope():
            initializer = NullInitializer(self._key_dtype, self._value_dtype)
            return tf.lookup.StaticHashTable(initializer, self._default_value)

    def _lookup_table_from_tokens(self, tokens):
        with tf.init_scope():
            token_start = self._token_start_index()
            token_end = token_start + tf.size(tokens)
            indices_dtype = self._key_dtype if self.invert else self._value_dtype
            indices = tf.range(token_start, token_end, dtype=indices_dtype)
            keys, values = (indices, tokens) if self.invert else (tokens, indices)
            initializer = tf.lookup.KeyValueTensorInitializer(keys, values, self._key_dtype, self._value_dtype)
            return tf.lookup.StaticHashTable(initializer, self._default_value)

    def _lookup_table_from_file(self, filename):
        if self.invert:
            key_index = tf.lookup.TextFileIndex.LINE_NUMBER
            value_index = tf.lookup.TextFileIndex.WHOLE_LINE
        else:
            key_index = tf.lookup.TextFileIndex.WHOLE_LINE
            value_index = tf.lookup.TextFileIndex.LINE_NUMBER
        with tf.init_scope():
            initializer = tf.lookup.TextFileInitializer(filename=filename, key_dtype=self._key_dtype, key_index=key_index, value_dtype=self._value_dtype, value_index=value_index, value_index_offset=self._token_start_index())
            return tf.lookup.StaticHashTable(initializer, self._default_value)

    def _convert_to_ndarray(self, x):
        return np.array(x) if isinstance(x, (list, tuple)) else x

    def _expand_dims(self, inputs, axis):
        if tf_utils.is_sparse(inputs):
            return tf.sparse.expand_dims(inputs, axis)
        else:
            return tf.expand_dims(inputs, axis)

    def _oov_start_index(self):
        return 1 if self.mask_token is not None and self.output_mode == INT else 0

    def _token_start_index(self):
        return self._oov_start_index() + self.num_oov_indices

    def _ensure_known_vocab_size(self):
        if self.output_mode == INT or self.pad_to_max_tokens:
            return
        if self._frozen_vocab_size is None:
            raise RuntimeError(f"When using `output_mode={self.output_mode}` and `pad_to_max_tokens=False`, you must set the layer's vocabulary before calling it. Either pass a `vocabulary` argument to the layer, or call `adapt` with some sample data.".format(self.output_mode))

    def _ensure_vocab_size_unchanged(self):
        if self.output_mode == INT or self.pad_to_max_tokens:
            return
        with tf.init_scope():
            new_vocab_size = self.vocabulary_size()
        if self._frozen_vocab_size is not None and new_vocab_size != self._frozen_vocab_size:
            raise RuntimeError(f'When using `output_mode={self.output_mode}` and `pad_to_max_tokens=False`, the vocabulary size cannot be changed after the layer is called. Old vocab size is {self._frozen_vocab_size}, new vocab size is {new_vocab_size}')

    def _find_repeated_tokens(self, vocabulary):
        """Return all repeated tokens in a vocabulary."""
        vocabulary_set = set(vocabulary)
        if len(vocabulary) != len(vocabulary_set):
            return [item for item, count in collections.Counter(vocabulary).items() if count > 1]
        else:
            return []

    def _num_tokens(self, data):
        """Count the number of tokens in a ragged, sparse or dense tensor."""
        if tf_utils.is_sparse(data):
            flat_values = data.values
        elif tf_utils.is_ragged(data):
            flat_values = data.flat_values
        else:
            flat_values = tf.reshape(data, [-1])
        tokens, _, counts = tf.unique_with_counts(flat_values, out_idx=tf.int64)
        return (tokens, counts)

    def _inverse_document_frequency(self, token_document_counts, num_documents):
        """Computes the inverse-document-frequency (IDF) component of "tf_idf".

        Uses the default weighting scheme described in
        https://en.wikipedia.org/wiki/Tf%E2%80%93idf.

        Args:
          token_document_counts: An array of the # of documents each token
            appears in.
          num_documents: An int representing the total number of documents

        Returns:
          An array of "inverse document frequency" weights.
        """
        return tf.math.log(1 + num_documents / (1 + token_document_counts))

    @property
    def _trackable_saved_model_saver(self):
        return layer_serialization.VocabularySavedModelSaver(self)

    def _tensor_vocab_to_numpy(self, vocabulary):
        """Converts a tensor vocabulary to a numpy vocabulary."""
        return vocabulary.numpy()