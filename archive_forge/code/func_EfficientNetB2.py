import copy
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.efficientnet.EfficientNetB2', 'keras.applications.EfficientNetB2')
def EfficientNetB2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax', **kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3, model_name='efficientnetb2', include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes, classifier_activation=classifier_activation, **kwargs)