import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import utils
from keras.src.applications import imagenet_utils
from keras.src.engine import sequential
from keras.src.engine import training as training_lib
from tensorflow.python.util.tf_export import keras_export
def ConvNeXt(depths, projection_dims, drop_path_rate=0.0, layer_scale_init_value=1e-06, default_size=224, model_name='convnext', include_preprocessing=True, include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    """Instantiates ConvNeXt architecture given specific configuration.

    Args:
      depths: An iterable containing depths for each individual stages.
      projection_dims: An iterable containing output number of channels of
      each individual stages.
      drop_path_rate: Stochastic depth probability. If 0.0, then stochastic
        depth won't be used.
      layer_scale_init_value: Layer scale coefficient. If 0.0, layer scaling
        won't be used.
      default_size: Default input image size.
      model_name: An optional name for the model.
      include_preprocessing: boolean denoting whther to include preprocessing in
        the model. When `weights="imagenet"` this should be always set to True.
        But for other models (e.g., randomly initialized) users should set it
        to False and apply preprocessing to data accordingly.
      include_top: Boolean denoting whether to include classification head to
        the model.
      weights: one of `None` (random initialization), `"imagenet"` (pre-training
        on ImageNet-1k), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top`
        is False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor output
          of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output
          of the last convolutional layer, and thus the output of the model will
          be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax`, or `None`
          when using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000
          when using ImageNet.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and (classes != 1000):
        raise ValueError("If using `weights` as `'imagenet'` with `include_top` as true, `classes` should be 1000")
    input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=default_size, min_size=32, data_format=backend.image_data_format(), require_flatten=include_top, weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    elif not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
    if input_tensor is not None:
        inputs = utils.layer_utils.get_source_inputs(input_tensor)[0]
    else:
        inputs = img_input
    x = inputs
    if include_preprocessing:
        channel_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        num_channels = input_shape[channel_axis - 1]
        if num_channels == 3:
            x = PreStem(name=model_name)(x)
    stem = sequential.Sequential([layers.Conv2D(projection_dims[0], kernel_size=4, strides=4, name=model_name + '_stem_conv'), layers.LayerNormalization(epsilon=1e-06, name=model_name + '_stem_layernorm')], name=model_name + '_stem')
    downsample_layers = []
    downsample_layers.append(stem)
    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = sequential.Sequential([layers.LayerNormalization(epsilon=1e-06, name=model_name + '_downsampling_layernorm_' + str(i)), layers.Conv2D(projection_dims[i + 1], kernel_size=2, strides=2, name=model_name + '_downsampling_conv_' + str(i))], name=model_name + '_downsampling_block_' + str(i))
        downsample_layers.append(downsample_layer)
    depth_drop_rates = [float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))]
    cur = 0
    num_convnext_blocks = 4
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtBlock(projection_dim=projection_dims[i], drop_path_rate=depth_drop_rates[cur + j], layer_scale_init_value=layer_scale_init_value, name=model_name + f'_stage_{i}_block_{j}')(x)
        cur += depths[i]
    if include_top:
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = Head(num_classes=classes, classifier_activation=classifier_activation, name=model_name)(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        x = layers.LayerNormalization(epsilon=1e-06)(x)
    model = training_lib.Model(inputs=inputs, outputs=x, name=model_name)
    if weights == 'imagenet':
        if include_top:
            file_suffix = '.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_suffix = '_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        file_name = model_name + file_suffix
        weights_path = utils.data_utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model