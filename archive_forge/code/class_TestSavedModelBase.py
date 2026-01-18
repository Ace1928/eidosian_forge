import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
class TestSavedModelBase(tf.test.TestCase, parameterized.TestCase):
    """Base class for testing saving/loading with DS."""

    def setUp(self):
        np.random.seed(_RANDOM_SEED)
        tf.compat.v1.set_random_seed(_RANDOM_SEED)
        self._root_dir = 'base'
        super().setUp()

    def _save_model(self, model, saved_dir):
        """Save the given model to the given saved_dir.

        This method needs to be implemented by the subclasses.

        Args:
          model: a keras model object to save.
          saved_dir: a string representing the path to save the keras model
        """
        raise NotImplementedError('must be implemented in descendants')

    def _load_and_run_model(self, distribution, saved_dir, predict_dataset, output_name='output_1'):
        """Load the model and run 1 step of predict with it.

        This method must be implemented by the subclasses.

        Args:
          distribution: the distribution strategy used to load the model. None
            if no distribution strategy is used
          saved_dir: the string representing the path where the model is saved.
          predict_dataset: the data used to do the predict on the model for
            cross_replica context.
          output_name: the string representing the name of the output layer of
            the model.
        """
        raise NotImplementedError('must be implemented in descendants')

    def _train_model(self, model, x_train, y_train, batch_size):
        training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        training_dataset = training_dataset.repeat()
        training_dataset = training_dataset.batch(batch_size)
        model.fit(x=training_dataset, epochs=1, steps_per_epoch=100)

    def _predict_with_model(self, distribution, model, predict_dataset):
        return model.predict(predict_dataset, steps=PREDICT_STEPS)

    def _get_predict_dataset(self, x_predict, batch_size):
        predict_dataset = tf.data.Dataset.from_tensor_slices(x_predict)
        predict_dataset = predict_dataset.repeat()
        predict_dataset = predict_dataset.batch(batch_size)
        return predict_dataset

    def run_test_save_no_strategy_restore_strategy(self, model_and_input, distribution):
        """Save a model without DS, and restore it with DS."""
        saved_dir = os.path.join(self.get_temp_dir(), '0')
        model = model_and_input.get_model()
        x_train, y_train, x_predict = model_and_input.get_data()
        batch_size = model_and_input.get_batch_size()
        predict_dataset = self._get_predict_dataset(x_predict, batch_size)
        self._train_model(model, x_train, y_train, batch_size)
        result_before_save = self._predict_with_model(None, model, predict_dataset)
        self._save_model(model, saved_dir)
        with distribution.scope():
            result_after_save = self._load_and_run_model(distribution=distribution, saved_dir=saved_dir, predict_dataset=predict_dataset)
        self.assertAllClose(result_before_save, result_after_save)

    def run_test_save_strategy_restore_no_strategy(self, model_and_input, distribution, save_in_scope):
        """Save a model with DS, and restore it without DS."""
        saved_dir = os.path.join(self.get_temp_dir(), '1')
        with distribution.scope():
            model = model_and_input.get_model()
            x_train, y_train, x_predict = model_and_input.get_data()
            batch_size = model_and_input.get_batch_size()
            self._train_model(model, x_train, y_train, batch_size)
            predict_dataset = self._get_predict_dataset(x_predict, batch_size)
            result_before_save = self._predict_with_model(distribution, model, predict_dataset)
        if save_in_scope:
            with distribution.scope():
                self._save_model(model, saved_dir)
        else:
            self._save_model(model, saved_dir)
        load_result = self._load_and_run_model(distribution=None, saved_dir=saved_dir, predict_dataset=predict_dataset)
        self.assertAllClose(result_before_save, load_result)

    def run_test_save_strategy_restore_strategy(self, model_and_input, distribution_for_saving, distribution_for_restoring, save_in_scope):
        """Save a model with DS, and restore it with potentially different
        DS."""
        saved_dir = os.path.join(self.get_temp_dir(), '2')
        with distribution_for_saving.scope():
            model = model_and_input.get_model()
            x_train, y_train, x_predict = model_and_input.get_data()
            batch_size = model_and_input.get_batch_size()
            self._train_model(model, x_train, y_train, batch_size)
            predict_dataset = self._get_predict_dataset(x_predict, batch_size)
            result_before_save = self._predict_with_model(distribution_for_saving, model, predict_dataset)
        if save_in_scope:
            with distribution_for_saving.scope():
                self._save_model(model, saved_dir)
        else:
            self._save_model(model, saved_dir)
        with distribution_for_restoring.scope():
            load_result = self._load_and_run_model(distribution=distribution_for_restoring, saved_dir=saved_dir, predict_dataset=predict_dataset)
        self.assertAllClose(result_before_save, load_result)

    def run_test_save_strategy(self, model_and_input, distribution, save_in_scope):
        """Save a model with DS."""
        saved_dir = os.path.join(self.get_temp_dir(), '3')
        with distribution.scope():
            model = model_and_input.get_model()
            x_train, y_train, _ = model_and_input.get_data()
            batch_size = model_and_input.get_batch_size()
            self._train_model(model, x_train, y_train, batch_size)
        if save_in_scope:
            with distribution.scope():
                self._save_model(model, saved_dir)
        else:
            self._save_model(model, saved_dir)
        return saved_dir