from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class BaseDNNWarmStartingTest(object):

    def __init__(self, _dnn_classifier_fn, _dnn_regressor_fn, fc_impl=feature_column):
        self._dnn_classifier_fn = _dnn_classifier_fn
        self._dnn_regressor_fn = _dnn_regressor_fn
        self._fc_impl = fc_impl

    def setUp(self):
        self._ckpt_and_vocab_dir = tempfile.mkdtemp()

        def _input_fn():
            features = {'city': [['Palo Alto'], ['Mountain View']], 'locality': [['Palo Alto'], ['Mountain View']], 'occupation': [['doctor'], ['consultant']]}
            return (features, [0, 1])
        self._input_fn = _input_fn

    def tearDown(self):
        tf.compat.v1.summary.FileWriterCache.clear()
        shutil.rmtree(self._ckpt_and_vocab_dir)

    def assertAllNotClose(self, t1, t2):
        """Helper assert for arrays."""
        sum_of_abs_diff = 0.0
        for x, y in zip(t1, t2):
            try:
                for a, b in zip(x, y):
                    sum_of_abs_diff += abs(b - a)
            except TypeError:
                sum_of_abs_diff += abs(y - x)
        self.assertGreater(sum_of_abs_diff, 0)

    def test_classifier_basic_warm_starting(self):
        """Tests correctness of DNNClassifier default warm-start."""
        city = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_list('city', vocabulary_list=['Mountain View', 'Palo Alto']), dimension=5)
        dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[city], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
        dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        warm_started_dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[city], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=dnn_classifier.model_dir)
        warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        for variable_name in warm_started_dnn_classifier.get_variable_names():
            self.assertAllClose(dnn_classifier.get_variable_value(variable_name), warm_started_dnn_classifier.get_variable_value(variable_name))

    def test_regressor_basic_warm_starting(self):
        """Tests correctness of DNNRegressor default warm-start."""
        city = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_list('city', vocabulary_list=['Mountain View', 'Palo Alto']), dimension=5)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=[256, 128], feature_columns=[city], model_dir=self._ckpt_and_vocab_dir, optimizer='SGD')
        dnn_regressor.train(input_fn=self._input_fn, max_steps=1)
        warm_started_dnn_regressor = self._dnn_regressor_fn(hidden_units=[256, 128], feature_columns=[city], optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=dnn_regressor.model_dir)
        warm_started_dnn_regressor.train(input_fn=self._input_fn, max_steps=1)
        for variable_name in warm_started_dnn_regressor.get_variable_names():
            self.assertAllClose(dnn_regressor.get_variable_value(variable_name), warm_started_dnn_regressor.get_variable_value(variable_name))

    def test_warm_starting_selective_variables(self):
        """Tests selecting variables to warm-start."""
        city = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_list('city', vocabulary_list=['Mountain View', 'Palo Alto']), dimension=5)
        dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[city], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
        dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        warm_started_dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[city], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=dnn_classifier.model_dir, vars_to_warm_start='.*(city).*'))
        warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        for variable_name in warm_started_dnn_classifier.get_variable_names():
            if 'city' in variable_name:
                self.assertAllClose(dnn_classifier.get_variable_value(variable_name), warm_started_dnn_classifier.get_variable_value(variable_name))
            elif 'bias' in variable_name:
                bias_values = warm_started_dnn_classifier.get_variable_value(variable_name)
                self.assertAllClose(np.zeros_like(bias_values), bias_values)
            elif 'kernel' in variable_name:
                self.assertAllNotClose(dnn_classifier.get_variable_value(variable_name), warm_started_dnn_classifier.get_variable_value(variable_name))

    def test_warm_starting_with_vocab_remapping_and_partitioning(self):
        """Tests warm-starting with vocab remapping and partitioning."""
        vocab_list = ['doctor', 'lawyer', 'consultant']
        vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'occupation_vocab')
        with open(vocab_file, 'w') as f:
            f.write('\n'.join(vocab_list))
        occupation = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_file('occupation', vocabulary_file=vocab_file, vocabulary_size=len(vocab_list)), dimension=2)
        partitioner = tf.compat.v1.fixed_size_partitioner(num_shards=2)
        dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[occupation], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD', input_layer_partitioner=partitioner)
        dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        new_vocab_list = ['doctor', 'consultant', 'engineer']
        new_vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'new_occupation_vocab')
        with open(new_vocab_file, 'w') as f:
            f.write('\n'.join(new_vocab_list))
        new_occupation = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_file('occupation', vocabulary_file=new_vocab_file, vocabulary_size=len(new_vocab_list)), dimension=2)
        occupation_vocab_info = estimator.VocabInfo(new_vocab=new_occupation.categorical_column.vocabulary_file, new_vocab_size=new_occupation.categorical_column.vocabulary_size, num_oov_buckets=new_occupation.categorical_column.num_oov_buckets, old_vocab=occupation.categorical_column.vocabulary_file, old_vocab_size=occupation.categorical_column.vocabulary_size, backup_initializer=tf.compat.v1.initializers.random_uniform(minval=0.39, maxval=0.39))
        warm_started_dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[occupation], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=dnn_classifier.model_dir, var_name_to_vocab_info={OCCUPATION_EMBEDDING_NAME: occupation_vocab_info}, vars_to_warm_start=None), input_layer_partitioner=partitioner)
        warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        self.assertAllClose(dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[0, :], warm_started_dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[0, :])
        self.assertAllClose(dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[2, :], warm_started_dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[1, :])
        self.assertAllClose([0.39] * 2, warm_started_dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[2, :])
        for variable_name in warm_started_dnn_classifier.get_variable_names():
            if 'bias' in variable_name:
                bias_values = warm_started_dnn_classifier.get_variable_value(variable_name)
                self.assertAllClose(np.zeros_like(bias_values), bias_values)
            elif 'kernel' in variable_name:
                self.assertAllNotClose(dnn_classifier.get_variable_value(variable_name), warm_started_dnn_classifier.get_variable_value(variable_name))

    def test_warm_starting_with_naming_change(self):
        """Tests warm-starting with a Tensor name remapping."""
        locality = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_list('locality', vocabulary_list=['Mountain View', 'Palo Alto']), dimension=5)
        dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[locality], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
        dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        city = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_list('city', vocabulary_list=['Mountain View', 'Palo Alto']), dimension=5)
        warm_started_dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[city], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=dnn_classifier.model_dir, var_name_to_prev_var_name={CITY_EMBEDDING_NAME: CITY_EMBEDDING_NAME.replace('city', 'locality')}))
        warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
        for variable_name in warm_started_dnn_classifier.get_variable_names():
            if 'city' in variable_name:
                self.assertAllClose(dnn_classifier.get_variable_value(CITY_EMBEDDING_NAME.replace('city', 'locality')), warm_started_dnn_classifier.get_variable_value(CITY_EMBEDDING_NAME))
            else:
                self.assertAllClose(dnn_classifier.get_variable_value(variable_name), warm_started_dnn_classifier.get_variable_value(variable_name))