import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import List
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import PYTHON_VERSION, insecure_hash
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.utils.requirements_utils import (
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
from mlflow.version import VERSION
class _PythonEnv:
    BUILD_PACKAGES = ('pip', 'setuptools', 'wheel')

    def __init__(self, python=None, build_dependencies=None, dependencies=None):
        """
        Represents environment information for MLflow Models and Projects.

        Args:
            python: Python version for the environment. If unspecified, defaults to the current
                Python version.
            build_dependencies: List of build dependencies for the environment that must
                be installed before installing ``dependencies``. If unspecified,
                defaults to an empty list.
            dependencies: List of dependencies for the environment. If unspecified, defaults to
                an empty list.
        """
        if python is not None and (not isinstance(python, str)):
            raise TypeError(f'`python` must be a string but got {type(python)}')
        if build_dependencies is not None and (not isinstance(build_dependencies, list)):
            raise TypeError(f'`build_dependencies` must be a list but got {type(build_dependencies)}')
        if dependencies is not None and (not isinstance(dependencies, list)):
            raise TypeError(f'`dependencies` must be a list but got {type(dependencies)}')
        self.python = python or PYTHON_VERSION
        self.build_dependencies = build_dependencies or []
        self.dependencies = dependencies or []

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def current(cls):
        return cls(python=PYTHON_VERSION, build_dependencies=cls.get_current_build_dependencies(), dependencies=[f'-r {_REQUIREMENTS_FILE_NAME}'])

    @staticmethod
    def _get_package_version(package_name):
        try:
            return __import__(package_name).__version__
        except (ImportError, AttributeError, AssertionError):
            return None

    @staticmethod
    def get_current_build_dependencies():
        build_dependencies = []
        for package in _PythonEnv.BUILD_PACKAGES:
            version = _PythonEnv._get_package_version(package)
            dep = package + '==' + version if version else package
            build_dependencies.append(dep)
        return build_dependencies

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_yaml(self, path):
        with open(path, 'w') as f:
            data = {k: v for k, v in self.to_dict().items() if v}
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @staticmethod
    def get_dependencies_from_conda_yaml(path):
        with open(path) as f:
            conda_env = yaml.safe_load(f)
        python = None
        build_dependencies = None
        unmatched_dependencies = []
        dependencies = None
        for dep in conda_env.get('dependencies', []):
            if isinstance(dep, str):
                match = _CONDA_DEPENDENCY_REGEX.match(dep)
                if not match:
                    unmatched_dependencies.append(dep)
                    continue
                package = match.group('package')
                operator = match.group('operator')
                version = match.group('version')
                if not python and package == 'python':
                    if operator is None:
                        raise MlflowException.invalid_parameter_value(f'Invalid dependency for python: {dep}. It must be pinned (e.g. python=3.8.13).')
                    if operator in ('<', '>', '!='):
                        raise MlflowException(f"Invalid version comparator for python: '{operator}'. Must be one of ['<=', '>=', '=', '=='].", error_code=INVALID_PARAMETER_VALUE)
                    python = version
                    continue
                if build_dependencies is None:
                    build_dependencies = []
                operator = '==' if operator == '=' else operator
                build_dependencies.append(package + (operator or '') + (version or ''))
            elif _is_pip_deps(dep):
                dependencies = dep['pip']
            else:
                raise MlflowException(f'Invalid conda dependency: {dep}. Must be str or dict in the form of {{"pip": [...]}}', error_code=INVALID_PARAMETER_VALUE)
        if python is None:
            _logger.warning(f'{path} does not include a python version specification. Using the current python version {PYTHON_VERSION}.')
            python = PYTHON_VERSION
        if unmatched_dependencies:
            _logger.warning('The following conda dependencies will not be installed in the resulting environment: %s', unmatched_dependencies)
        return {'python': python, 'build_dependencies': build_dependencies, 'dependencies': dependencies}

    @classmethod
    def from_conda_yaml(cls, path):
        return cls.from_dict(cls.get_dependencies_from_conda_yaml(path))