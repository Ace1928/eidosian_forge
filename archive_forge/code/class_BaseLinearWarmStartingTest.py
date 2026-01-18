from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
class BaseLinearWarmStartingTest(object):

    def __init__(self, _linear_classifier_fn, _linear_regressor_fn, fc_lib=feature_column):
        self._linear_classifier_fn = _linear_classifier_fn
        self._linear_regressor_fn = _linear_regressor_fn
        self._fc_lib = fc_lib

    def setUp(self):
        self._ckpt_and_vocab_dir = tempfile.mkdtemp()

        def _input_fn():
            features = {'age': [[23.0], [31.0]], 'age_in_years': [[23.0], [31.0]], 'occupation': [['doctor'], ['consultant']]}
            return (features, [0, 1])
        self._input_fn = _input_fn

    def tearDown(self):
        tf.compat.v1.summary.FileWriterCache.clear()
        shutil.rmtree(self._ckpt_and_vocab_dir)

    def test_classifier_basic_warm_starting(self):
        """Tests correctness of LinearClassifier default warm-start."""
        age = self._fc_lib.numeric_column('age')
        linear_classifier = self._linear_classifier_fn(feature_columns=[age], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
        linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        warm_started_linear_classifier = self._linear_classifier_fn(feature_columns=[age], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=linear_classifier.model_dir)
        warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        for variable_name in warm_started_linear_classifier.get_variable_names():
            self.assertAllClose(linear_classifier.get_variable_value(variable_name), warm_started_linear_classifier.get_variable_value(variable_name))

    def test_regressor_basic_warm_starting(self):
        """Tests correctness of LinearRegressor default warm-start."""
        age = self._fc_lib.numeric_column('age')
        linear_regressor = self._linear_regressor_fn(feature_columns=[age], model_dir=self._ckpt_and_vocab_dir, optimizer='SGD')
        linear_regressor.train(input_fn=self._input_fn, max_steps=1)
        warm_started_linear_regressor = self._linear_regressor_fn(feature_columns=[age], optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=linear_regressor.model_dir)
        warm_started_linear_regressor.train(input_fn=self._input_fn, max_steps=1)
        for variable_name in warm_started_linear_regressor.get_variable_names():
            self.assertAllClose(linear_regressor.get_variable_value(variable_name), warm_started_linear_regressor.get_variable_value(variable_name))

    def test_warm_starting_selective_variables(self):
        """Tests selecting variables to warm-start."""
        age = self._fc_lib.numeric_column('age')
        linear_classifier = self._linear_classifier_fn(feature_columns=[age], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
        linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        warm_started_linear_classifier = self._linear_classifier_fn(feature_columns=[age], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=linear_classifier.model_dir, vars_to_warm_start='.*(age).*'))
        warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        self.assertAllClose(linear_classifier.get_variable_value(AGE_WEIGHT_NAME), warm_started_linear_classifier.get_variable_value(AGE_WEIGHT_NAME))
        self.assertAllClose([0.0] * 4, warm_started_linear_classifier.get_variable_value(BIAS_NAME))

    def test_warm_starting_with_vocab_remapping_and_partitioning(self):
        """Tests warm-starting with vocab remapping and partitioning."""
        vocab_list = ['doctor', 'lawyer', 'consultant']
        vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'occupation_vocab')
        with open(vocab_file, 'w') as f:
            f.write('\n'.join(vocab_list))
        occupation = self._fc_lib.categorical_column_with_vocabulary_file('occupation', vocabulary_file=vocab_file, vocabulary_size=len(vocab_list))
        partitioner = tf.compat.v1.fixed_size_partitioner(num_shards=2)
        linear_classifier = self._linear_classifier_fn(feature_columns=[occupation], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD', partitioner=partitioner)
        linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        new_vocab_list = ['doctor', 'consultant', 'engineer']
        new_vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'new_occupation_vocab')
        with open(new_vocab_file, 'w') as f:
            f.write('\n'.join(new_vocab_list))
        new_occupation = self._fc_lib.categorical_column_with_vocabulary_file('occupation', vocabulary_file=new_vocab_file, vocabulary_size=len(new_vocab_list))
        occupation_vocab_info = estimator.VocabInfo(new_vocab=new_occupation.vocabulary_file, new_vocab_size=new_occupation.vocabulary_size, num_oov_buckets=new_occupation.num_oov_buckets, old_vocab=occupation.vocabulary_file, old_vocab_size=occupation.vocabulary_size, backup_initializer=tf.compat.v1.initializers.random_uniform(minval=0.39, maxval=0.39))
        warm_started_linear_classifier = self._linear_classifier_fn(feature_columns=[occupation], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=linear_classifier.model_dir, var_name_to_vocab_info={OCCUPATION_WEIGHT_NAME: occupation_vocab_info}, vars_to_warm_start=None), partitioner=partitioner)
        warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        self.assertAllClose(linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[0, :], warm_started_linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[0, :])
        self.assertAllClose(linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[2, :], warm_started_linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[1, :])
        self.assertAllClose([0.39] * 4, warm_started_linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[2, :])
        self.assertAllClose([0.0] * 4, warm_started_linear_classifier.get_variable_value(BIAS_NAME))

    def test_warm_starting_with_naming_change(self):
        """Tests warm-starting with a Tensor name remapping."""
        age_in_years = self._fc_lib.numeric_column('age_in_years')
        linear_classifier = self._linear_classifier_fn(feature_columns=[age_in_years], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
        linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        warm_started_linear_classifier = self._linear_classifier_fn(feature_columns=[self._fc_lib.numeric_column('age')], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=linear_classifier.model_dir, var_name_to_prev_var_name={AGE_WEIGHT_NAME: AGE_WEIGHT_NAME.replace('age', 'age_in_years')}))
        warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
        self.assertAllClose(linear_classifier.get_variable_value(AGE_WEIGHT_NAME.replace('age', 'age_in_years')), warm_started_linear_classifier.get_variable_value(AGE_WEIGHT_NAME))
        self.assertAllClose(linear_classifier.get_variable_value(BIAS_NAME), warm_started_linear_classifier.get_variable_value(BIAS_NAME))