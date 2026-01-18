import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.regnet.RegNetX008', 'keras.applications.RegNetX008')
def RegNetX008(model_name='regnetx008', include_top=True, include_preprocessing=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    return RegNet(MODEL_CONFIGS['x008']['depths'], MODEL_CONFIGS['x008']['widths'], MODEL_CONFIGS['x008']['group_width'], MODEL_CONFIGS['x008']['block_type'], MODEL_CONFIGS['x008']['default_size'], model_name=model_name, include_top=include_top, include_preprocessing=include_preprocessing, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes, classifier_activation=classifier_activation)