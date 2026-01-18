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
def EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate=0.2, drop_connect_rate=0.2, depth_divisor=8, activation='swish', blocks_args='default', model_name='efficientnet', include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    """Instantiates the EfficientNet architecture.

    Args:
      width_coefficient: float, scaling coefficient for network width.
      depth_coefficient: float, scaling coefficient for network depth.
      default_size: integer, default input image size.
      dropout_rate: float, dropout rate before final classifier layer.
      drop_connect_rate: float, dropout rate at skip connections.
      depth_divisor: integer, a unit of network width.
      activation: activation function.
      blocks_args: list of dicts, parameters to construct block modules.
      model_name: string, model name.
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False.
          It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.

    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and (classes != 1000):
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000')
    input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=default_size, min_size=32, data_format=backend.image_data_format(), require_flatten=include_top, weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    elif not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))
    x = img_input
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Normalization(axis=bn_axis)(x)
    if weights == 'imagenet':
        x = layers.Rescaling([1.0 / math.sqrt(stddev) for stddev in IMAGENET_STDDEV_RGB])(x)
    x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, 3), name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32), 3, strides=2, padding='valid', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum((round_repeats(args['repeats']) for args in blocks_args)))
    for i, args in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])
        for j in range(round_repeats(args.pop('repeats'))):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation, drop_connect_rate * b / blocks, name=f'block{i + 1}{chr(j + 97)}_', **args)
            b += 1
    x = layers.Conv2D(round_filters(1280), 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, kernel_initializer=DENSE_KERNEL_INITIALIZER, name='predictions')(x)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = training.Model(inputs, x, name=model_name)
    if weights == 'imagenet':
        if include_top:
            file_suffix = '.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        else:
            file_suffix = '_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suffix
        weights_path = data_utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model