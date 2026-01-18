from typing import Dict
from typing import List
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import adapters
from autokeras import analysers
from autokeras import blocks
from autokeras import hyper_preprocessors as hpps_module
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module
class TimeseriesInput(StructuredDataInput):
    """Input node for timeseries data.

    # Arguments
        lookback: Int. The range of history steps to consider for each prediction.
            For example, if lookback=n, the data in the range of [i - n, i - 1]
            is used to predict the value of step i. If unspecified, it will be tuned
            automatically.
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data. A column will be judged as
            categorical if the number of different values is less than 5% of the
            number of instances.
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, lookback: Optional[int]=None, column_names: Optional[List[str]]=None, column_types: Optional[Dict[str, str]]=None, name: Optional[str]=None, **kwargs):
        super().__init__(column_names=column_names, column_types=column_types, name=name, **kwargs)
        self.lookback = lookback

    def get_config(self):
        config = super().get_config()
        config.update({'lookback': self.lookback})
        return config

    def get_adapter(self):
        return adapters.TimeseriesAdapter()

    def get_analyser(self):
        return analysers.TimeseriesAnalyser(column_names=self.column_names, column_types=self.column_types)

    def get_block(self):
        return blocks.TimeseriesBlock()

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self.column_names:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.CategoricalToNumericalPreprocessor(column_names=self.column_names, column_types=self.column_types)))
        hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.SlidingWindow(lookback=self.lookback, batch_size=self.batch_size)))
        return hyper_preprocessors