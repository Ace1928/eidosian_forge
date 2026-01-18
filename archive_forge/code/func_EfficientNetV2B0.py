import copy
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.efficientnet_v2.EfficientNetV2B0', 'keras.applications.EfficientNetV2B0')
def EfficientNetV2B0(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax', include_preprocessing=True):
    return EfficientNetV2(width_coefficient=1.0, depth_coefficient=1.0, default_size=224, model_name='efficientnetv2-b0', include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes, classifier_activation=classifier_activation, include_preprocessing=include_preprocessing)