from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import pandas as pd
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tasks import structured_data
from autokeras.tuners import greedy
from autokeras.utils import types
class SupervisedTimeseriesDataPipeline(structured_data.BaseStructuredDataPipeline):

    def __init__(self, outputs, column_names=None, column_types=None, lookback=None, predict_from=1, predict_until=None, **kwargs):
        inputs = input_module.TimeseriesInput(lookback=lookback, column_names=column_names, column_types=column_types)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.predict_from = predict_from
        self.predict_until = predict_until
        self._target_col_name = None
        self.train_len = 0

    @staticmethod
    def _read_from_csv(x, y):
        df = pd.read_csv(x)
        target = df.pop(y).dropna().to_numpy()
        return (df, target)

    def fit(self, x=None, y=None, epochs=None, callbacks=None, validation_split=0.2, validation_data=None, **kwargs):
        if isinstance(x, str):
            self._target_col_name = y
            x, y = self._read_from_csv(x, y)
        if validation_data:
            x_val, y_val = validation_data
            if isinstance(x_val, str):
                validation_data = self._read_from_csv(x_val, y_val)
        self.check_in_fit(x)
        self.train_len = len(y)
        if validation_data:
            x_val, y_val = validation_data
            train_len = len(y_val)
            x_val = x_val[:train_len]
            y_val = y_val[self.lookback - 1:]
            validation_data = (x_val, y_val)
        history = super().fit(x=x[:self.train_len], y=y[self.lookback - 1:], epochs=epochs, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, **kwargs)
        return history

    def predict(self, x, **kwargs):
        x = self.read_for_predict(x)
        if len(x) < self.train_len:
            raise ValueError('The prediction data requires the original training data to make predictions on subsequent data points')
        y_pred = super().predict(x=x, **kwargs)
        lower_bound = self.train_len + self.predict_from
        if self.predict_until is None:
            self.predict_until = len(y_pred)
        upper_bound = min(self.train_len + self.predict_until + 1, len(y_pred))
        return y_pred[lower_bound:upper_bound]

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
        return super().evaluate(x=x[:len(y)], y=y[self.lookback - 1:], **kwargs)