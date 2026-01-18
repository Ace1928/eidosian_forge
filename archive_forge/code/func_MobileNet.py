import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.mobilenet.MobileNet', 'keras.applications.MobileNet')
def MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000, classifier_activation='softmax', **kwargs):
    """Instantiates the MobileNet architecture.

    Reference:
    - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNet, call `tf.keras.applications.mobilenet.preprocess_input`
    on your inputs before passing them to the model.
    `mobilenet.preprocess_input` will scale input pixels between -1 and 1.

    Args:
      input_shape: Optional shape tuple, only to be specified if `include_top`
        is False (otherwise the input shape has to be `(224, 224, 3)` (with
        `channels_last` data format) or (3, 224, 224) (with `channels_first`
        data format). It should have exactly 3 inputs channels, and width and
        height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
        valid value. Defaults to `None`.
        `input_shape` will be ignored if the `input_tensor` is provided.
      alpha: Controls the width of the network. This is known as the width
        multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
        decreases the number of filters in each layer. - If `alpha` > 1.0,
        proportionally increases the number of filters in each layer. - If
        `alpha` = 1, default number of filters from the paper are used at each
        layer. Defaults to `1.0`.
      depth_multiplier: Depth multiplier for depthwise convolution. This is
        called the resolution multiplier in the MobileNet paper.
        Defaults to `1.0`.
      dropout: Dropout rate. Defaults to `0.001`.
      include_top: Boolean, whether to include the fully-connected layer at the
        top of the network. Defaults to `True`.
      weights: One of `None` (random initialization), 'imagenet' (pre-training
        on ImageNet), or the path to the weights file to be loaded. Defaults to
        `imagenet`.
      input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model. `input_tensor` is useful for sharing
        inputs between multiple different networks. Defaults to `None`.
      pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` (default) means that the output of the model will be
            the 4D tensor output of the last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: Optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified. Defaults to `1000`.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    """
    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError(f'Unknown argument(s): {(kwargs,)}')
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError(f'The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.  Received weights={weights}')
    if weights == 'imagenet' and include_top and (classes != 1000):
        raise ValueError(f'If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000.  Received classes={classes}')
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]
        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224
    input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=default_size, min_size=32, data_format=backend.image_data_format(), require_flatten=include_top, weights=weights)
    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError(f'If imagenet weights are being loaded, depth multiplier must be 1.  Received depth_multiplier={depth_multiplier}')
        if alpha not in [0.25, 0.5, 0.75, 1.0]:
            raise ValueError(f'If imagenet weights are being loaded, alpha can be one of`0.25`, `0.50`, `0.75` or `1.0` only.  Received alpha={alpha}')
        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            logging.warning('`input_shape` is undefined or non-square, or `rows` is not in [128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.')
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    elif not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = layers.Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(activation=classifier_activation, name='predictions')(x)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = training.Model(inputs, x, name=f'mobilenet_{alpha:0.2f}_{rows}')
    if weights == 'imagenet':
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.5:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'
        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(model_name, weight_path, cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model