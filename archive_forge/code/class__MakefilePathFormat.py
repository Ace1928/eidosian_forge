import hashlib
import logging
import os
import pathlib
import re
import shutil
from typing import Dict, List
from mlflow.environment_variables import (
from mlflow.recipes.step import BaseStep, StepStatus
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd
class _MakefilePathFormat:
    """
    Provides platform-agnostic path substitution for execution Makefiles, ensuring that POSIX-style
    relative paths are joined correctly with POSIX-style or Windows-style recipe root paths.

    For example, given a format string `s = "{path:prp/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat(recipe_root_path="/my/recipe/root/path", ...))` on
    Unix systems or
    `s.format(path=_MakefilePathFormat(recipe_root_path="C:\\my\\recipe\\root\\path", ...))`` on
    Windows systems will yield "/my/recipe/root/path/my/subpath.txt" or
    "C:/my/recipe/root/path/my/subpath.txt", respectively.

    Additionally, given a format string `s = "{path:exe/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat(execution_directory_path="/my/exe/dir/path", ...))` on
    Unix systems or
    `s.format(path=_MakefilePathFormat(execution_directory_path="/my/exe/dir/path", ...))`` on
    Windows systems will yield "/my/exe/dir/path/my/subpath.txt" or
    "C:/my/exe/dir/path/my/subpath.txt", respectively.
    """
    _RECIPE_ROOT_PATH_PREFIX_PLACEHOLDER = 'prp/'
    _EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER = 'exe/'

    def __init__(self, recipe_root_path: str, execution_directory_path: str):
        """
        Args:
            recipe_root_path: The absolute path of the recipe root directory on the local
                filesystem.
            execution_directory_path: The absolute path of the execution directory on the local
                filesystem for the recipe.
        """
        self.recipe_root_path = recipe_root_path
        self.execution_directory_path = execution_directory_path

    def _get_formatted_path(self, path_spec: str, prefix_placeholder: str, replacement_path: str) -> str:
        """
        Args:
            path_spec: A substitution path spec of the form `<placeholder>/<subpath>`. This
                method substitutes `<placeholder>` with `<recipe_root_path>`, if
                `<placeholder>` is `prp`, or `<execution_directory_path>`, if
                `<placeholder>` is `exe`.
            prefix_placeholder: The prefix placeholder, which is present at the beginning of
                `path_spec`. Either `prp` or `exe`.
            replacement_path: The path to use to replace the specified `prefix_placeholder`
                in the specified `path_spec`.

        Returns:
            The formatted path obtained by replacing the ``prefix placeholder`` in the
            specified ``path_spec`` with the specified ``replacement_path``.
        """
        subpath = pathlib.PurePosixPath(path_spec.split(prefix_placeholder)[1])
        recipe_root_posix_path = pathlib.PurePosixPath(pathlib.Path(replacement_path).as_posix())
        full_formatted_path = recipe_root_posix_path / subpath
        return str(full_formatted_path)

    def __format__(self, path_spec: str) -> str:
        """
        Args:
            path_spec: A substitution path spec of the form `<placeholder>/<subpath>`. This
                method substitutes `<placeholder>` with `<recipe_root_path>`, if
                `<placeholder>` is `prp`, or `<execution_directory_path>`, if
                `<placeholder>` is `exe`.
        """
        if path_spec.startswith(_MakefilePathFormat._RECIPE_ROOT_PATH_PREFIX_PLACEHOLDER):
            return self._get_formatted_path(path_spec=path_spec, prefix_placeholder=_MakefilePathFormat._RECIPE_ROOT_PATH_PREFIX_PLACEHOLDER, replacement_path=self.recipe_root_path)
        elif path_spec.startswith(_MakefilePathFormat._EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER):
            return self._get_formatted_path(path_spec=path_spec, prefix_placeholder=_MakefilePathFormat._EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER, replacement_path=self.execution_directory_path)
        else:
            raise ValueError(f'Invalid Makefile string format path spec: {path_spec}')