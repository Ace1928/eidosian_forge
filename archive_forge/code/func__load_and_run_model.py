import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from keras.src.distribute import model_combinations
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