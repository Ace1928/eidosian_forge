import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def DenseNet(blocks, include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    """Instantiates the DenseNet architecture.

    Reference:
    - [Densely Connected Convolutional Networks](
        https://arxiv.org/abs/1608.06993) (CVPR 2017)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of input preprocessing.
    For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your
    inputs before passing them to the model.
    `densenet.preprocess_input` will scale pixels between 0 and 1 and then
    will normalize each channel with respect to the ImageNet dataset statistics.

    Args:
      blocks: numbers of building blocks for the four dense layers.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `'channels_last'` data format)
        or `(3, 224, 224)` (with `'channels_first'` data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and (classes != 1000):
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000')
    input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=224, min_size=32, data_format=backend.image_data_format(), require_flatten=include_top, weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    elif not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    if blocks == [6, 12, 24, 16]:
        model = training.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = training.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = training.Model(inputs, x, name='densenet201')
    else:
        model = training.Model(inputs, x, name='densenet')
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file('densenet121_weights_tf_dim_ordering_tf_kernels.h5', DENSENET121_WEIGHT_PATH, cache_subdir='models', file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file('densenet169_weights_tf_dim_ordering_tf_kernels.h5', DENSENET169_WEIGHT_PATH, cache_subdir='models', file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file('densenet201_weights_tf_dim_ordering_tf_kernels.h5', DENSENET201_WEIGHT_PATH, cache_subdir='models', file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        elif blocks == [6, 12, 24, 16]:
            weights_path = data_utils.get_file('densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET121_WEIGHT_PATH_NO_TOP, cache_subdir='models', file_hash='30ee3e1110167f948a6b9946edeeb738')
        elif blocks == [6, 12, 32, 32]:
            weights_path = data_utils.get_file('densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET169_WEIGHT_PATH_NO_TOP, cache_subdir='models', file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
        elif blocks == [6, 12, 48, 32]:
            weights_path = data_utils.get_file('densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET201_WEIGHT_PATH_NO_TOP, cache_subdir='models', file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model