import inspect
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import jax_utils
from keras.src.utils import tracking
from keras.src.utils import tree
from keras.src.utils.module_utils import jax
@keras_export('keras.layers.FlaxLayer')
class FlaxLayer(JaxLayer):
    """Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.

    This layer enables the use of Flax components in the form of
    [`flax.linen.Module`](
        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)
    instances within Keras when using JAX as the backend for Keras.

    The module method to use for the forward pass can be specified via the
    `method` argument and is `__call__` by default. This method must take the
    following arguments with these exact names:

    - `self` if the method is bound to the module, which is the case for the
        default of `__call__`, and `module` otherwise to pass the module.
    - `inputs`: the inputs to the model, a JAX array or a `PyTree` of arrays.
    - `training` *(optional)*: an argument specifying if we're in training mode
        or inference mode, `True` is passed in training mode.

    `FlaxLayer` handles the non-trainable state of your model and required RNGs
    automatically. Note that the `mutable` parameter of
    [`flax.linen.Module.apply()`](
        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply)
    is set to `DenyList(["params"])`, therefore making the assumption that all
    the variables outside of the "params" collection are non-trainable weights.

    This example shows how to create a `FlaxLayer` from a Flax `Module` with
    the default `__call__` method and no training argument:

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, inputs):
            x = inputs
            x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(flax_module)
    ```

    This example shows how to wrap the module method to conform to the required
    signature. This allows having multiple input arguments and a training
    argument that has a different name and values. This additionally shows how
    to use a function that is not bound to the module.

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def forward(self, input1, input2, deterministic):
            ...
            return outputs

    def my_flax_module_wrapper(module, inputs, training):
        input1, input2 = inputs
        return module.forward(input1, input2, not training)

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(
        module=flax_module,
        method=my_flax_module_wrapper,
    )
    ```

    Args:
        module: An instance of `flax.linen.Module` or subclass.
        method: The method to call the model. This is generally a method in the
            `Module`. If not provided, the `__call__` method is used. `method`
            can also be a function not defined in the `Module`, in which case it
            must take the `Module` as the first argument. It is used for both
            `Module.init` and `Module.apply`. Details are documented in the
            `method` argument of [`flax.linen.Module.apply()`](
              https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply).
        variables: A `dict` containing all the variables of the module in the
            same format as what is returned by [`flax.linen.Module.init()`](
              https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.init).
            It should contain a "params" key and, if applicable, other keys for
            collections of variables for non-trainable state. This allows
            passing trained parameters and learned non-trainable state or
            controlling the initialization. If `None` is passed, the module's
            `init` function is called at build time to initialize the variables
            of the model.
    """

    def __init__(self, module, method=None, variables=None, **kwargs):
        from flax.core import scope as flax_scope
        if backend.backend() != 'jax':
            raise ValueError(f'FlaxLayer is only supported with the JAX backend. Current backend: {backend.backend()}')
        self.module = module
        self.method = method
        apply_mutable = flax_scope.DenyList(['params'])

        def apply_with_training(params, state, rng, inputs, training):
            return self.module.apply(self._params_and_state_to_variables(params, state), inputs, rngs=rng, method=self.method, mutable=apply_mutable, training=training)

        def apply_without_training(params, state, rng, inputs):
            return self.module.apply(self._params_and_state_to_variables(params, state), inputs, rngs=rng, method=self.method, mutable=apply_mutable)

        def init_with_training(rng, inputs, training):
            return self._variables_to_params_and_state(self.module.init(rng, inputs, method=self.method, training=training))

        def init_without_training(rng, inputs):
            return self._variables_to_params_and_state(self.module.init(rng, inputs, method=self.method))
        if 'training' in inspect.signature(method or module.__call__).parameters:
            call_fn, init_fn = (apply_with_training, init_with_training)
        else:
            call_fn, init_fn = (apply_without_training, init_without_training)
        params, state = self._variables_to_params_and_state(variables)
        super().__init__(call_fn=call_fn, init_fn=init_fn, params=params, state=state, **kwargs)

    def _params_and_state_to_variables(self, params, state):
        if params:
            if state:
                return {**params, **state}
            else:
                return params
        elif state:
            return state
        return {}

    def _variables_to_params_and_state(self, variables):
        if variables is None:
            return (None, None)
        if 'params' not in variables:
            return ({}, variables)
        if len(variables) == 1:
            return (variables, {})
        params = {'params': variables['params']}
        state = {k: v for k, v in variables.items() if k != 'params'}
        return (params, state)

    def _get_init_rng(self):
        return {'params': self.seed_generator.next(), 'dropout': self.seed_generator.next()}

    def _get_call_rng(self, training):
        if training:
            return {'dropout': self.seed_generator.next()}
        else:
            return {}

    def get_config(self):
        config_method = self.method
        if hasattr(self.method, '__self__') and self.method.__self__ == self.module:
            config_method = self.method.__name__
        config = {'module': serialization_lib.serialize_keras_object(self.module), 'method': serialization_lib.serialize_keras_object(config_method)}
        base_config = super().get_config()
        base_config.pop('call_fn')
        base_config.pop('init_fn')
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        module = serialization_lib.deserialize_keras_object(config['module'])
        method = serialization_lib.deserialize_keras_object(config['method'])
        if isinstance(config['method'], str):
            method = getattr(module, method)
        config['module'] = module
        config['method'] = method
        return cls(**config)