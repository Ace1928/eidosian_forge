import hashlib
import os
import asyncio
import aiofiles
from inspect import getsource, getmembers, isfunction, isclass, iscoroutinefunction
from Cython.Build import cythonize
from anyio import Path
from setuptools import setup, Extension
import sys
import logging
import logging.config
import pathlib
from typing import (
from indelogging import (
import concurrent_log_handler
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import (
from inspect import iscoroutinefunction
from functools import wraps
class CythonCompiler:

    def __init__(self, obj: Type, module_name: str, hash_file: FILE_PATH):
        self.obj = obj
        self.module_name = module_name
        self.hash_file = hash_file
        self.logger = logging.getLogger(__name__)
        ensure_logging_config_exists(LOGGING_CONF=FILES['LOGGING_CONF'])
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

    @staticmethod
    def _generate_hash() -> str:
        """
        Generates a unique hash for the current state of the source code of the object.
        This hash is used to determine if the source code has changed since the last compilation.
        """
        source = getsource(self.obj)
        return hashlib.sha256(source.encode('utf-8')).hexdigest()

    async def _file_exists_and_matches(self, file_path: FILE_PATH, current_hash: str) -> bool:
        """
        Checks if the hash file exists and if its content matches the current hash.
        """
        if not file_path.exists():
            return False
        async with aiofiles.open(file_path, mode='r') as file:
            stored_hash = await file.read()
        return stored_hash == current_hash

    async def _write_to_file(self, file_path: FILE_PATH, content: str) -> None:
        """
        Writes the given content to the specified file.
        """
        async with aiofiles.open(file_path, mode='w') as file:
            await file.write(content)

    def _compile_cython(self) -> None:
        """
        Compiles the Python source code to a Cython C extension.
        """
        setup(ext_modules=cythonize(Extension(name=self.module_name, sources=[getsource(self.obj)])))

    def _execute_c(self) -> None:
        """
        Dynamically imports and executes the compiled C extension.
        """
        try:
            __import__(self.module_name)
        except ImportError as e:
            self.logger.error(f'Failed to import module {self.module_name}: {e}')

    @UniversalDecorator()
    async def ensure_latest_version_and_execute(self) -> None:
        """
        Ensures the latest version of the Python object is compiled to a Cython C extension and executes it.
        """
        current_hash = self._generate_hash()
        if await self._file_exists_and_matches(self.hash_file, current_hash):
            self.logger.info(f'No changes detected. Executing C version of {self.obj.__name__}.')
            self._execute_c()
        else:
            self.logger.info(f'Changes detected or C version not found. Updating {self.obj.__name__}.')
            self._compile_cython()
            await self._write_to_file(self.hash_file, current_hash)
            self._execute_c()