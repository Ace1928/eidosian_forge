import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
class BaseOptimizer:

    def __init__(self, learning_rate, weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None, name=None, **kwargs):
        self._lock = False
        if kwargs.pop('decay', None) is not None:
            warnings.warn('Argument `decay` is no longer supported and will be ignored.')
        if kwargs:
            raise ValueError(f'Argument(s) not recognized: {kwargs}')
        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.clipvalue = clipvalue
        self.use_ema = use_ema
        self.loss_scale_factor = loss_scale_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if gradient_accumulation_steps:
            if not gradient_accumulation_steps >= 2:
                raise ValueError(f'`gradient_accumulation_steps` must be an integer >= 2. Received: gradient_accumulation_steps={gradient_accumulation_steps}')
        if use_ema:
            if ema_momentum > 1 or ema_momentum < 0:
                raise ValueError(f'`ema_momentum` must be in the range [0, 1]. Received: ema_momentum={ema_momentum}')
            if ema_overwrite_frequency and (not isinstance(ema_overwrite_frequency, int) or ema_overwrite_frequency < 1):
                raise ValueError(f'`ema_overwrite_frequency` must be an integer >= 1 or None. Received: ema_overwrite_frequency={ema_overwrite_frequency}')
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency
        if self.clipnorm is not None and self.global_clipnorm is not None:
            raise ValueError(f'Only one of `clipnorm` and `global_clipnorm` can be set. Received: clipnorm={self.clipnorm}, global_clipnorm={self.global_clipnorm}')
        self.built = False
        self._variables = []
        self._trainable_variables = []
        self._tracker = tracking.Tracker({'variables': (lambda x: isinstance(x, backend.Variable), self._variables)})
        self._trainable_variables_indices = {}
        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(0, name='iteration', dtype='int', trainable=False)
        self._track_variable(iterations)
        self.iterations = iterations
        if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if not isinstance(learning_rate, float):
                raise ValueError(f'Argument `learning_rate` should be float, or an instance of LearningRateSchedule, or a callable (that takes in the current iteration value and returns the corresponding learning rate value). Received instead: learning_rate={learning_rate}')
            with backend.name_scope(self.name, caller=self):
                learning_rate = backend.Variable(learning_rate, name='learning_rate', dtype=backend.floatx(), trainable=False)
            self._track_variable(learning_rate)
            self._learning_rate = learning_rate

    def _track_variable(self, variable):
        self._tracker.add_to_store('variables', variable)

    @tracking.no_automatic_dependency_tracking
    def build(self, variables):
        if self.use_ema:
            self._model_variables_moving_average = []
        if self.gradient_accumulation_steps:
            self._accumulated_gradients = []
        for i, variable in enumerate(variables):
            self._trainable_variables_indices[self._var_key(variable)] = i
            if self.use_ema:
                self._model_variables_moving_average.append(self.add_variable_from_reference(variable, name='average'))
            if self.gradient_accumulation_steps:
                self._accumulated_gradients.append(self.add_variable_from_reference(variable, name='gradient_accumulator'))
        self._trainable_variables = variables[:]
        self.built = True

    def _var_key(self, variable):
        return id(variable)

    @property
    def variables(self):
        return self._variables[:]

    def _get_variable_index(self, variable):
        return self._trainable_variables_indices[self._var_key(variable)]

    def add_variable(self, shape, initializer='zeros', dtype=None, name=None):
        self._check_super_called()
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(initializer=initializer, shape=shape, dtype=dtype, trainable=False, name=name)
        self._track_variable(variable)
        return variable

    def add_variable_from_reference(self, reference_variable, name=None, initializer='zeros'):
        """Add an all-zeros variable with the shape and dtype of a reference
        variable.
        """
        name = name or 'var'
        if hasattr(reference_variable, 'path'):
            name = reference_variable.path.replace('/', '_') + '_' + name
        else:
            name = str(reference_variable.name).replace(':', '_') + '_' + name
        return self.add_variable(shape=reference_variable.shape, initializer=initializer, dtype=reference_variable.dtype, name=name)

    def _check_variables_are_known(self, variables):
        for v in variables:
            if self._var_key(v) not in self._trainable_variables_indices:
                raise ValueError(f'Unknown variable: {v}. This optimizer can only be called for the variables it was originally built with. When working with a new set of variables, you should recreate a new optimizer instance.')

    def assign(self, variable, value):
        """Assign a value to a variable.

        This should be used in optimizers instead of `variable.assign(value)` to
        support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign(value)

    def assign_add(self, variable, value):
        """Add a value to a variable.

        This should be used in optimizers instead of
        `variable.assign_add(value)` to support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign_add(value)

    def assign_sub(self, variable, value):
        """Subtract a value from a variable.

        This should be used in optimizers instead of
        `variable.assign_sub(value)` to support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign_sub(value)

    def update_step(self, gradient, variable, learning_rate):
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        return self.iterations

    def apply(self, grads, trainable_variables=None):
        """Update traininable variables according to provided gradient values.

        `grads` should be a list of gradient tensors
        with 1:1 mapping to the list of variables the optimizer was built with.

        `trainable_variables` can be provided
        on the first call to build the optimizer.
        """
        if len(grads) == 0:
            return
        if trainable_variables is None:
            if not self.built:
                raise ValueError('When passing `grads` without `variables`, the optimizer must already be built on a list of variables. Call `optimizer.build(trainable_variables)` first. ')
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(f'When passing `grads` as a list of gradient tensors, the gradients must match `optimizer.variables` one-to-on. Received a list of {len(grads)} gradients, but the optimizer is tracking {len(self._trainable_variables)} trainable variables.')
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            if not self.built:
                with backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)
        with backend.name_scope(self.name, caller=self):
            grads, trainable_variables = self._filter_empty_gradients(grads, trainable_variables)
            if len(list(grads)) == 0:
                return
            scale = self.loss_scale_factor
            if scale is not None:
                grads = [g if g is None else g / scale for g in grads]
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)
            self._backend_apply_gradients(grads, trainable_variables)
            for variable in trainable_variables:
                if getattr(variable, 'constraint', None) is not None:
                    variable.assign(variable.constraint(variable))

    def _backend_apply_gradients(self, grads, trainable_variables):
        """Apply method that can be overridden by different backends.

        JAX overrides it in order to deal with statelessness in gradient
        accumulation and EMA handling.

        The below implementation is intended to be generally backend-agnostic,
        but may not work with all backends.

        This method does 4 things:
        - Call the optimizer's update_step() to update trainable variables
            and optimizer variables.
        - Update EMA variables, if EMA is configured.
        - Update gradient accumulators, if gradient accumulation is configured.
        - Update the iteration counter.
        """
        if self.gradient_accumulation_steps:
            is_update_step = (self.iterations + 1) % self.gradient_accumulation_steps == 0

            def _update_step_fn(self, grads, trainable_variables):
                steps = self.gradient_accumulation_steps
                grads = [(grads[i] + self._accumulated_gradients[i]) / steps for i in range(len(grads))]
                self._backend_update_step(grads, trainable_variables, self.learning_rate)
                self._backend_reset_gradient_accumulators()

            def _grad_accumulation_fn(self, grads):
                self._backend_increment_gradient_accumulators(grads)
            ops.cond(is_update_step, lambda: _update_step_fn(self, grads, trainable_variables), lambda: _grad_accumulation_fn(self, grads))
        else:
            self._backend_update_step(grads, trainable_variables, self.learning_rate)
        if self.use_ema:
            self._update_model_variables_moving_average(self._trainable_variables)
            if self.ema_overwrite_frequency:
                should_overwrite_model_vars = (self.iterations + 1) % self.ema_overwrite_frequency == 0
                ops.cond(should_overwrite_model_vars, lambda: self._overwrite_model_variables_with_average_value(self._trainable_variables), lambda: None)
        self.iterations.assign_add(1)

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        for grad, var in zip(grads, trainable_variables):
            self.update_step(grad, var, learning_rate)

    def _backend_reset_gradient_accumulators(self):
        for g_acc in self._accumulated_gradients:
            g_acc.assign(np.zeros(g_acc.shape, dtype=g_acc.dtype))

    def _backend_increment_gradient_accumulators(self, grads):
        new_g_accs = [grads[i] + self._accumulated_gradients[i] for i in range(len(grads))]
        for n_g_acc, g_acc in zip(new_g_accs, self._accumulated_gradients):
            g_acc.assign(n_g_acc)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        self._check_super_called()
        if not self.built:
            raise ValueError(f'To call `stateless_apply`, {self.__class__.__name__} must be built (i.e. its variables must have been created). You can build it via `optimizer.build(trainable_variables)`.')
        if len(optimizer_variables) != len(self.variables):
            raise ValueError(f'Argument `optimizer_variables` must be a list of tensors corresponding 1:1 to {self.__class__.__name__}().variables. Received list with length {len(optimizer_variables)}, but expected {len(self.variables)} variables.')
        if len(trainable_variables) != len(self._trainable_variables):
            raise ValueError(f'Argument `optimizer_variables` must be a list of tensors corresponding 1:1 to the trainable variables list that the optimizer was built with. Received len(trainable_variables) == {len(trainable_variables)} whereas the optimizer was built with {len(self._trainable_variables)} variables.')
        mapping = list(zip(self._trainable_variables, trainable_variables)) + list(zip(self.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.apply(grads)
        trainable_variables = []
        for v in self._trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                trainable_variables.append(new_v)
            else:
                trainable_variables.append(v)
        optimizer_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                optimizer_variables.append(new_v)
            else:
                optimizer_variables.append(v)
        return (trainable_variables, optimizer_variables)

    def scale_loss(self, loss):
        """Scale the loss before computing gradients.

        Scales the loss before gradients are computed in a `train_step`. This
        is primarily useful during mixed precision training to prevent numeric
        underflow.
        """
        if self.loss_scale_factor is not None:
            return loss * self.loss_scale_factor
        return loss

    @property
    def learning_rate(self):
        return self._get_current_learning_rate()

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
                raise TypeError('This optimizer was created with a `LearningRateSchedule` object as its `learning_rate` constructor argument, hence its learning rate is not settable. If you need the learning rate to be settable, you should instantiate the optimizer with a float `learning_rate` argument.')
            self._learning_rate.assign(learning_rate)

    def set_weights(self, weights):
        """Set the weights of the optimizer."""
        if not self.built:
            raise ValueError('You are calling `set_weights()` on an optimizer that has not yet been built. Please call `optimizer.build(trainable_variables)` to create the optimizer weights before calling `set_weights()`.')
        for variable, weight in zip(self._variables, weights):
            if variable.shape != weight.shape:
                raise ValueError(f'Optimizer variable {self._var_key(variable)} has shape {str(variable.shape)} not compatible with provided weight shape {str(weight.shape)}.')
            variable.assign(weight)

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        """Set the state of this optimizer object."""
        if len(store.keys()) != len(self.variables):
            msg = f"Skipping variable loading for optimizer '{self.name}', because it has {len(self.variables)} variables whereas the saved optimizer has {len(store.keys())} variables. "
            if len(self.variables) == 0:
                msg += 'This is likely because the optimizer has not been called/built yet.'
            warnings.warn(msg, stacklevel=2)
            return
        for i, variable in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _get_current_learning_rate(self):
        if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
            return self._learning_rate(self.iterations)
        elif callable(self._learning_rate):
            return self._learning_rate(self.iterations)
        return self._learning_rate

    def _filter_empty_gradients(self, grads, vars):
        for grad in grads:
            if grad is None:
                filtered = [(g, v) for g, v in zip(grads, vars) if g is not None]
                if not filtered:
                    raise ValueError('No gradients provided for any variable.')
                if len(filtered) < len(grads):
                    missing_grad_vars = [v for g, v in zip(grads, vars) if g is None]
                    warnings.warn(f'Gradients do not exist for variables {[v.name for v in missing_grad_vars]} when minimizing the loss. If using `model.compile()`, did you forget to provide a `loss` argument?')
                return zip(*filtered)
        return (grads, vars)

    def _clip_gradients(self, grads):
        if self.clipnorm and self.clipnorm > 0:
            clipped_grads = []
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(self._clip_by_norm(g))
            return clipped_grads
        if self.global_clipnorm and self.global_clipnorm > 0:
            return clip_by_global_norm(grads, self.global_clipnorm)
        if self.clipvalue and self.clipvalue > 0:
            clipped_grads = []
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(ops.clip(g, -self.clipvalue, self.clipvalue))
            return clipped_grads
        return grads

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decay.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, '_built') and self._built:
            raise ValueError('`exclude_from_weight_decay()` can only be configued before the optimizer is built.')
        if var_list:
            self._exclude_from_weight_decay = [self._var_key(variable) for variable in var_list]
        else:
            self._exclude_from_weight_decay = []
        self._exclude_from_weight_decay_names = var_names or []

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(self, '_exclude_from_weight_decay', [])
        exclude_from_weight_decay_names = getattr(self, '_exclude_from_weight_decay_names', [])
        variable_id = self._var_key(variable)
        for exclude_id in exclude_from_weight_decay:
            if variable_id == exclude_id:
                return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = ops.cast(self.learning_rate, variable.dtype)
                wd = ops.cast(self.weight_decay, variable.dtype)
                variable.assign(variable - variable * wd * lr)

    def _check_super_called(self):
        if not hasattr(self, '_lock'):
            raise RuntimeError(f"In optimizer '{self.__class__.__name__}', you forgot to call `super().__init__()` as the first statement in the `__init__()` method. Go add it!")

    def _update_model_variables_moving_average(self, trainable_variables):
        """Update the stored moving average using the latest value."""
        if self.use_ema:
            for var, average in zip(trainable_variables, self._model_variables_moving_average):
                not_first_step = ops.not_equal(self.iterations, 0)
                momentum = ops.cast(not_first_step, var.dtype) * self.ema_momentum
                average.assign(momentum * average + (1 - momentum) * var)

    def _overwrite_model_variables_with_average_value(self, trainable_variables):
        """Overwrite model variables with its moving average."""
        if len(trainable_variables) != len(self._model_variables_moving_average):
            raise ValueError(f'The length of model variables ({len(trainable_variables)}) to override does not match the length of model variables stored in the optimizer ({len(self._model_variables_moving_average)}). Please check if the optimizer was called on your model.')
        for var, average_var in zip(trainable_variables, self._model_variables_moving_average):
            var.assign(average_var)

    def finalize_variable_values(self, var_list):
        """Set the final value of model's trainable variables.

        Sometimes there are some extra steps before ending the variable updates,
        such as overriding the model variables with its average value.

        Args:
          var_list: list of model variables.
        """
        if self.use_ema:
            self._overwrite_model_variables_with_average_value(var_list)

    def get_config(self):
        """Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Subclass optimizer should override this method to include other
        hyperparameters.

        Returns:
            Python dictionary.
        """
        if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
            learning_rate = learning_rate_schedule.serialize(self._learning_rate)
        elif isinstance(self._learning_rate, backend.Variable):
            learning_rate = float(self._learning_rate.numpy())
        elif ops.is_tensor(self._learning_rate):
            learning_rate = float(self._learning_rate)
        elif callable(self._learning_rate):
            learning_rate = serialization_lib.serialize_keras_object(self._learning_rate)
        config = {'name': self.name, 'learning_rate': learning_rate, 'weight_decay': self.weight_decay, 'clipnorm': self.clipnorm, 'global_clipnorm': self.global_clipnorm, 'clipvalue': self.clipvalue, 'use_ema': self.use_ema, 'ema_momentum': self.ema_momentum, 'ema_overwrite_frequency': self.ema_overwrite_frequency, 'loss_scale_factor': self.loss_scale_factor, 'gradient_accumulation_steps': self.gradient_accumulation_steps}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates an optimizer from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same optimizer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.
            custom_objects: A Python dictionary mapping names to additional
              user-defined Python objects needed to recreate this optimizer.

        Returns:
            An optimizer instance.
        """
        if 'learning_rate' in config:
            if isinstance(config['learning_rate'], dict):
                config['learning_rate'] = serialization_lib.deserialize_keras_object(config['learning_rate'], custom_objects=custom_objects)
        return cls(**config)

    def __setattr__(self, name, value):
        if name != '_lock':
            self._check_super_called()
        if hasattr(self, '_tracker'):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _clip_by_norm(self, values, axes=None):
        l2sum = ops.sum(ops.square(values), axes, keepdims=True)
        pred = l2sum > 0
        l2sum_safe = ops.where(pred, l2sum, ops.ones_like(l2sum))
        l2norm = ops.where(pred, ops.sqrt(l2sum_safe), l2sum)
        intermediate = ops.multiply(values, self.clipnorm)
        values_clip = ops.convert_to_tensor(intermediate) / ops.maximum(l2norm, self.clipnorm)
        return values_clip