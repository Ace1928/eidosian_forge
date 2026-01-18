from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
from keras_tuner.src.backend import ops
from keras_tuner.src.backend.keras import layers
from keras_tuner.src.engine import hypermodel
@keras_tuner_export('keras_tuner.applications.HyperResNet')
class HyperResNet(hypermodel.HyperModel):
    """A ResNet hypermodel.

    Models built by `HyperResNet` take images with shape (height, width,
    channels) as input. The output are one-hot encoded with the length matching
    the number of classes specified by the `classes` argument.

    Args:
        include_top: Boolean, whether to include the fully-connected layer at
            the top of the network.
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.  One of
            `input_shape` or `input_tensor` must be specified.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.  One of `input_shape` or
            `input_tensor` must be specified.
        classes: Optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        **kwargs: Additional keyword arguments that apply to all hypermodels.
            See `keras_tuner.HyperModel`.
    """

    def __init__(self, include_top=True, input_shape=None, input_tensor=None, classes=None, **kwargs):
        super().__init__(**kwargs)
        if include_top and classes is None:
            raise ValueError('You must specify `classes` when `include_top=True`')
        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` or `input_tensor`.')
        self.include_top = include_top
        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.classes = classes

    def build(self, hp):
        version = hp.Choice('version', ['v1', 'v2', 'next'], default='v2')
        conv3_depth = hp.Choice('conv3_depth', [4, 8])
        conv4_depth = hp.Choice('conv4_depth', [6, 23, 36])
        preact = version == 'v2'
        use_bias = version != 'next'
        bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
        if self.input_tensor is not None:
            inputs = keras.utils.get_source_inputs(self.input_tensor)
            x = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
        if not preact:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
        if version == 'v1':
            x = stack1(x, 64, 3, stride1=1, name='conv2')
            x = stack1(x, 128, conv3_depth, name='conv3')
            x = stack1(x, 256, conv4_depth, name='conv4')
            x = stack1(x, 512, 3, name='conv5')
        elif version == 'v2':
            x = stack2(x, 64, 3, name='conv2')
            x = stack2(x, 128, conv3_depth, name='conv3')
            x = stack2(x, 256, conv4_depth, name='conv4')
            x = stack2(x, 512, 3, stride1=1, name='conv5')
        elif version == 'next':
            x = stack3(x, 64, 3, name='conv2')
            x = stack3(x, 256, conv3_depth, name='conv3')
            x = stack3(x, 512, conv4_depth, name='conv4')
            x = stack3(x, 1024, 3, stride1=1, name='conv5')
        if preact:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name='post_bn')(x)
            x = layers.Activation('relu', name='post_relu')(x)
        pooling = hp.Choice('pooling', ['avg', 'max'], default='avg')
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
        if not self.include_top:
            return keras.Model(inputs, x, name='ResNet')
        x = layers.Dense(self.classes, activation='softmax', name='probs')(x)
        model = keras.Model(inputs, x, name='ResNet')
        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
        optimizer = keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001], default=0.01)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model