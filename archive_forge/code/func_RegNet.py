import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def RegNet(depths, widths, group_width, block_type, default_size, model_name='regnet', include_preprocessing=True, include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    """Instantiates RegNet architecture given specific configuration.

    Args:
      depths: An iterable containing depths for each individual stages.
      widths: An iterable containing output channel width of each individual
        stages
      group_width: Number of channels to be used in each group. See grouped
        convolutions for more information.
      block_type: Must be one of `{"X", "Y", "Z"}`. For more details see the
        papers "Designing network design spaces" and "Fast and Accurate Model
        Scaling"
      default_size: Default input image size.
      model_name: An optional name for the model.
      include_preprocessing: boolean denoting whther to include preprocessing in
        the model
      include_top: Boolean denoting whether to include classification head to
        the model.
      weights: one of `None` (random initialization), "imagenet" (pre-training
        on ImageNet), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top`
        is False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`. - `None` means that the output of the model will be the 4D
        tensor output of the last convolutional layer. - `avg` means that global
        average pooling will be applied to the output of the last convolutional
        layer, and thus the output of the model will be a 2D tensor. - `max`
        means that global max pooling will be applied.
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
        ValueError: if `classifier_activation` is not `softmax` or `None` when
          using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000.
        ValueError: if `block_type` is not one of `{"X", "Y", "Z"}`

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
        inputs = layer_utils.get_source_inputs(input_tensor)[0]
    else:
        inputs = img_input
    x = inputs
    if include_preprocessing:
        x = PreStem(name=model_name)(x)
    x = Stem(name=model_name)(x)
    in_channels = 32
    for num_stage in range(4):
        depth = depths[num_stage]
        out_channels = widths[num_stage]
        x = Stage(block_type, depth, group_width, in_channels, out_channels, name=model_name + '_Stage_' + str(num_stage))(x)
        in_channels = out_channels
    if include_top:
        x = Head(num_classes=classes)(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    model = training.Model(inputs=inputs, outputs=x, name=model_name)
    if weights == 'imagenet':
        if include_top:
            file_suffix = '.h5'
            file_hash = WEIGHTS_HASHES[model_name[-4:]][0]
        else:
            file_suffix = '_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name[-4:]][1]
        file_name = model_name + file_suffix
        weights_path = data_utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model