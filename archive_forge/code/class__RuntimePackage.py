import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
class _RuntimePackage:
    """Represents a Ray package loaded via ``load_package()``.

    This class provides access to the symbols defined by the interface file of
    the package (e.g., remote functions and actor definitions). You can also
    access the raw runtime env defined by the package via ``pkg._runtime_env``.
    """

    def __init__(self, name: str, desc: str, interface_file: str, runtime_env: dict):
        self._name = name
        self._description = desc
        self._interface_file = interface_file
        self._runtime_env = runtime_env
        _validate_interface_file(self._interface_file)
        spec = importlib.util.spec_from_file_location(self._name, self._interface_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        for symbol in dir(self._module):
            if not symbol.startswith('_'):
                value = getattr(self._module, symbol)
                if isinstance(value, ray.remote_function.RemoteFunction) or isinstance(value, ray.actor.ActorClass):
                    setattr(self, symbol, value.options(runtime_env=runtime_env))

    def __repr__(self):
        return 'ray._RuntimePackage(module={}, runtime_env={})'.format(self._module, self._runtime_env)