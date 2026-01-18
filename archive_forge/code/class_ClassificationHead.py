from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils
class ClassificationHead(head_module.Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.

    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, num_classes: Optional[int]=None, multi_label: bool=False, loss: Optional[types.LossType]=None, metrics: Optional[types.MetricsType]=None, dropout: Optional[float]=None, **kwargs):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.dropout = dropout
        if metrics is None:
            metrics = ['accuracy']
        if loss is None:
            loss = self.infer_loss()
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self._encoded = None
        self._encoded_for_sigmoid = None
        self._encoded_for_softmax = None
        self._add_one_dimension = False
        self._labels = None

    def infer_loss(self):
        if not self.num_classes:
            return None
        if self.num_classes == 2 or self.multi_label:
            return losses.BinaryCrossentropy()
        return losses.CategoricalCrossentropy()

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'multi_label': self.multi_label, 'dropout': self.dropout})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        if len(output_node.shape) > 2:
            output_node = reduction.SpatialReduction().build(hp, output_node)
        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice('dropout', [0.0, 0.25, 0.5], default=0)
        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        output_node = layers.Dense(self.shape[-1])(output_node)
        if isinstance(self.loss, keras.losses.BinaryCrossentropy):
            output_node = layers.Activation(activations.sigmoid, name=self.name)(output_node)
        else:
            output_node = layers.Softmax(name=self.name)(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ClassificationAdapter(name=self.name)

    def get_analyser(self):
        return analysers.ClassificationAnalyser(name=self.name, multi_label=self.multi_label)

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.num_classes = analyser.num_classes
        self.loss = self.infer_loss()
        self._encoded = analyser.encoded
        self._encoded_for_sigmoid = analyser.encoded_for_sigmoid
        self._encoded_for_softmax = analyser.encoded_for_softmax
        self._add_one_dimension = len(analyser.shape) == 1
        self._labels = analyser.labels

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self._add_one_dimension:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension()))
        if self.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.CastToInt32()))
        if not self._encoded and self.dtype != tf.string:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.CastToString()))
        if self._encoded_for_sigmoid:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.SigmoidPostprocessor()))
        elif self._encoded_for_softmax:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.SoftmaxPostprocessor()))
        elif self.num_classes == 2:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.LabelEncoder(self._labels)))
        else:
            hyper_preprocessors.append(hpps_module.DefaultHyperPreprocessor(preprocessors.OneHotEncoder(self._labels)))
        return hyper_preprocessors