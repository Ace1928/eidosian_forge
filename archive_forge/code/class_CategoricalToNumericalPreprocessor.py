import tensorflow as tf
from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils
class CategoricalToNumericalPreprocessor(preprocessor.Preprocessor):
    """Encode the categorical features to numerical features.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
    """

    def __init__(self, column_names, column_types, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types
        encoding = []
        for column_name in self.column_names:
            column_type = self.column_types[column_name]
            if column_type == analysers.CATEGORICAL:
                encoding.append(keras_layers.INT)
            else:
                encoding.append(keras_layers.NONE)
        self.layer = keras_layers.MultiCategoryEncoding(encoding)

    def fit(self, dataset):
        self.layer.adapt(dataset)

    def transform(self, dataset):
        return dataset.map(self.layer)

    def get_config(self):
        vocab = []
        for encoding_layer in self.layer.encoding_layers:
            if encoding_layer is None:
                vocab.append([])
            else:
                vocab.append(encoding_layer.get_vocabulary())
        return {'column_types': self.column_types, 'column_names': self.column_names, 'encoding_layer': preprocessors.serialize(self.layer), 'encoding_vocab': vocab}

    @classmethod
    def from_config(cls, config):
        init_config = {'column_types': config['column_types'], 'column_names': config['column_names']}
        obj = cls(**init_config)
        obj.layer = preprocessors.deserialize(config['encoding_layer'])
        for encoding_layer, vocab in zip(obj.layer.encoding_layers, config['encoding_vocab']):
            if encoding_layer is not None:
                encoding_layer.set_vocabulary(vocab)
        return obj