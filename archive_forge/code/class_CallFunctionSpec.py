import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
class CallFunctionSpec:
    """Caches the spec and provides utilities for handling call function
    args."""

    def __init__(self, full_argspec):
        """Initialies a `CallFunctionSpec`.

        Args:
          full_argspec: the FullArgSpec of a call function of a layer.
        """
        self._full_argspec = full_argspec
        self._arg_names = list(self._full_argspec.args)
        if self._arg_names and self._arg_names[0] == 'self':
            self._arg_names = self._arg_names[1:]
        self._arg_names += self._full_argspec.kwonlyargs or []
        call_accepts_kwargs = self._full_argspec.varkw is not None
        self._expects_training_arg = 'training' in self._arg_names or call_accepts_kwargs
        self._expects_mask_arg = 'mask' in self._arg_names or call_accepts_kwargs
        call_fn_defaults = self._full_argspec.defaults or []
        defaults = dict()
        for i in range(-1 * len(call_fn_defaults), 0):
            defaults[self._arg_names[i]] = call_fn_defaults[i]
        defaults.update(self._full_argspec.kwonlydefaults or {})
        self._default_training_arg = defaults.get('training')

    @property
    def full_argspec(self):
        """Returns the FullArgSpec of the call function."""
        return self._full_argspec

    @property
    def arg_names(self):
        """List of names of args and kwonlyargs."""
        return self._arg_names

    @arg_names.setter
    def arg_names(self, value):
        self._arg_names = value

    @property
    @cached_per_instance
    def arg_positions(self):
        """Returns a dict mapping arg names to their index positions."""
        call_fn_arg_positions = dict()
        for pos, arg in enumerate(self._arg_names):
            call_fn_arg_positions[arg] = pos
        return call_fn_arg_positions

    @property
    def expects_training_arg(self):
        """Whether the call function uses 'training' as a parameter."""
        return self._expects_training_arg

    @expects_training_arg.setter
    def expects_training_arg(self, value):
        self._expects_training_arg = value

    @property
    def expects_mask_arg(self):
        """Whether the call function uses `mask` as a parameter."""
        return self._expects_mask_arg

    @expects_mask_arg.setter
    def expects_mask_arg(self, value):
        self._expects_mask_arg = value

    @property
    def default_training_arg(self):
        """The default value given to the "training" argument."""
        return self._default_training_arg

    def arg_was_passed(self, arg_name, args, kwargs, inputs_in_args=False):
        """Returns true if argument is present in `args` or `kwargs`.

        Args:
          arg_name: String name of the argument to find.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.

        Returns:
          True if argument with `arg_name` is present in `args` or `kwargs`.
        """
        if not args and (not kwargs):
            return False
        if arg_name in kwargs:
            return True
        call_fn_args = self._arg_names
        if not inputs_in_args:
            call_fn_args = call_fn_args[1:]
        return arg_name in dict(zip(call_fn_args, args))

    def get_arg_value(self, arg_name, args, kwargs, inputs_in_args=False):
        """Retrieves the value for the argument with name `arg_name`.

        Args:
          arg_name: String name of the argument to find.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.

        Returns:
          The value of the argument with name `arg_name`, extracted from `args`
          or `kwargs`.

        Raises:
          KeyError if the value of `arg_name` cannot be found.
        """
        if arg_name in kwargs:
            return kwargs[arg_name]
        call_fn_args = self._arg_names
        if not inputs_in_args:
            call_fn_args = call_fn_args[1:]
        args_dict = dict(zip(call_fn_args, args))
        return args_dict[arg_name]

    def set_arg_value(self, arg_name, new_value, args, kwargs, inputs_in_args=False, pop_kwarg_if_none=False):
        """Sets the value of an argument into the given args/kwargs.

        Args:
          arg_name: String name of the argument to find.
          new_value: New value to give to the argument.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.
          pop_kwarg_if_none: If the new value is `None`, and this is `True`,
            then the argument is deleted from `kwargs`.

        Returns:
          The updated `(args, kwargs)`.
        """
        if self.full_argspec.varargs:
            try:
                arg_pos = self.full_argspec.args.index(arg_name)
                if self.full_argspec.args[0] == 'self':
                    arg_pos -= 1
            except ValueError:
                arg_pos = None
        else:
            arg_pos = self.arg_positions.get(arg_name, None)
        if arg_pos is not None:
            if not inputs_in_args:
                arg_pos = arg_pos - 1
            if len(args) > arg_pos:
                args = list(args)
                args[arg_pos] = new_value
                return (tuple(args), kwargs)
        if new_value is None and pop_kwarg_if_none:
            kwargs.pop(arg_name, None)
        else:
            kwargs[arg_name] = new_value
        return (args, kwargs)

    def split_out_first_arg(self, args, kwargs):
        """Splits (args, kwargs) into (inputs, args, kwargs)."""
        if args:
            inputs = args[0]
            args = args[1:]
        elif self._arg_names[0] in kwargs:
            kwargs = copy.copy(kwargs)
            inputs = kwargs.pop(self._arg_names[0])
        else:
            raise ValueError('The first argument to `Layer.call` must always be passed.')
        return (inputs, args, kwargs)