import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import pandas as pd
import tensorflow as tf
from tensorflow import nest
from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tuners import task_specific
from autokeras.utils import types
class BaseStructuredDataPipeline(auto_model.AutoModel):

    def __init__(self, inputs, outputs, **kwargs):
        self.check(inputs.column_names, inputs.column_types)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self._target_col_name = None

    @staticmethod
    def _read_from_csv(x, y):
        df = pd.read_csv(x)
        target = df.pop(y).to_numpy()
        return (df, target)

    def check(self, column_names, column_types):
        if column_types:
            for column_type in column_types.values():
                if column_type not in ['categorical', 'numerical']:
                    raise ValueError('column_types should be either "categorical" or "numerical", but got {name}'.format(name=column_type))

    def check_in_fit(self, x):
        input_node = nest.flatten(self.inputs)[0]
        if isinstance(x, pd.DataFrame) and input_node.column_names is None:
            input_node.column_names = list(x.columns)
        if input_node.column_names and input_node.column_types:
            for column_name in input_node.column_types:
                if column_name not in input_node.column_names:
                    raise ValueError('column_names and column_types are mismatched. Cannot find column name {name} in the data.'.format(name=column_name))

    def read_for_predict(self, x):
        if isinstance(x, str):
            x = pd.read_csv(x)
            if self._target_col_name in x:
                x.pop(self._target_col_name)
        return x

    def fit(self, x=None, y=None, epochs=None, callbacks=None, validation_split=0.2, validation_data=None, **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the training data.
            y: String, numpy.ndarray, or tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a string, which is the
                name of the target column. Otherwise, it can be single-column or
                multi-column. The values should all be numerical.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, we would use epochs equal to 1000 and early stopping
                with patience equal to 30.
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
                The best model found would be fit on the entire dataset including the
                validation data.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
                The best model found would be fit on the training dataset without the
                validation data.
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        # Returns
            history: A Keras History object corresponding to the best model.
                Its History.history attribute is a record of training
                loss values and metrics values at successive epochs, as well as
                validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, str):
            self._target_col_name = y
            x, y = self._read_from_csv(x, y)
        if validation_data and (not isinstance(validation_data, tf.data.Dataset)):
            x_val, y_val = validation_data
            if isinstance(x_val, str):
                validation_data = self._read_from_csv(x_val, y_val)
        self.check_in_fit(x)
        history = super().fit(x=x, y=y, epochs=epochs, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, **kwargs)
        return history

    def predict(self, x, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Testing data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the testing data.
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        x = self.read_for_predict(x)
        return super().predict(x=x, **kwargs)

    def evaluate(self, x, y=None, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Testing data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the testing data.
            y: String, numpy.ndarray, or tensorflow.Dataset. Testing data y.
                If the data is from a csv file, it should be a string corresponding
                to the label column.
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        if isinstance(x, str):
            x, y = self._read_from_csv(x, y)
        return super().evaluate(x=x, y=y, **kwargs)