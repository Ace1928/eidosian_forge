import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
class ConfigModule(ModuleType):
    _default: Dict[str, Any]
    _config: Dict[str, Any]
    _allowed_keys: Set[str]
    _bypass_keys: Set[str]

    def __init__(self):
        raise NotImplementedError(f'use {__name__}.install_config_module(sys.modules[__name__])')

    def __setattr__(self, name, value):
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._allowed_keys:
            raise AttributeError(f'{self.__name__}.{name} does not exist')
        else:
            self._config[name] = value

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f'{self.__name__}.{name} does not exist')

    def __delattr__(self, name):
        del self._config[name]

    def save_config(self):
        """Convert config to a pickled blob"""
        config = dict(self._config)
        for key in config.get('_save_config_ignore', ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def codegen_config(self):
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        lines = []
        mod = self.__name__
        for k, v in self._config.items():
            if k in self._config.get('_save_config_ignore', ()):
                continue
            if v == self._default[k]:
                continue
            lines.append(f'{mod}.{k} = {v!r}')
        return '\n'.join(lines)

    def load_config(self, data):
        """Restore from a prior call to save_config()"""
        self.to_dict().update(pickle.loads(data))

    def to_dict(self):
        return self._config

    def get_config_copy(self):
        return copy.deepcopy(self._config)

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                changes = {arg1: arg2}
            else:
                changes = arg1
            assert not kwargs
        else:
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f'expected `dict` got {type(changes)}'
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):

            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    prior[key] = config._config[key]
                config._config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config._config.update(prior)
                prior.clear()
        return ConfigPatch()