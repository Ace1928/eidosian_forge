import hashlib
import os
import asyncio
import aiofiles
from inspect import getsource, getmembers, isfunction, isclass
from Cython.Build import cythonize
from anyio import Path
from setuptools import setup, Extension
import sys
import logging
import logging.config
import pathlib
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Type,
    Dict,
    List,
    Tuple,
    TypeAlias,
    Union,
    Optional,
)
from indelogging import (
    DEFAULT_LOGGING_CONFIG,
    ensure_logging_config_exists,
    UniversalDecorator,
)
import concurrent_log_handler

# Comprehensive Type Aliases for Enhanced Code Readability and Consistency
# The following section defines a series of type aliases aimed at improving the readability and maintainability of the codebase. By providing explicit, descriptive types for commonly used concepts such as directory names, paths, file names, and file paths, these aliases facilitate a more intuitive and error-resistant development process. They leverage Python's typing system to ensure consistent usage of these types across the codebase, thereby enhancing the robustness and clarity of the code.

# Type alias for directory names. Expected to be string values representing the name of a directory.
DIR_NAME: TypeAlias = str
# Type alias for directory paths. Utilizes the pathlib.Path type for a clear, platform-independent representation of filesystem paths.
DIR_PATH: TypeAlias = pathlib.Path
# Type alias for file names. Similar to DIR_NAME, it represents file names as strings, aiding in distinguishing between general strings and file names.
FILE_NAME: TypeAlias = str
# Type alias for file paths. Employs pathlib.Path to represent paths to files, helping differentiate between directory paths (DIR_PATH) and file paths (FILE_PATH).
FILE_PATH: TypeAlias = pathlib.Path

# Immutable Codebase-Wide Constants
# The following dictionaries define constants for directories and files used throughout the codebase. These constants are not to be modified under any circumstances to ensure the integrity and consistency of file and directory references. They serve as a centralized point of reference for all directory and file paths, streamlining path management and reducing the risk of errors.

# Dictionary mapping directory names to their respective paths. Optional typing allows for the possibility of uninitialized paths.
DIRECTORIES: Dict[DIR_NAME, Optional[DIR_PATH]] = {
    "ROOT": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development"),
    "LOGS": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/logs"),
    "CONFIG": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config"
    ),
    "DATA": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/data"),
    "MEDIA": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/media"),
    "SCRIPTS": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/scripts"
    ),
    "TEMPLATES": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/templates"
    ),
    "UTILS": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/utils"),
}
# Dictionary mapping file names to their respective paths, ensuring that file paths are correctly typed and managed.
FILES: Dict[FILE_NAME, FILE_PATH] = {
    "DIRECTORIES_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/directories.conf"
    ),
    "FILES_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/files.conf"
    ),
    "DATABASE_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/database.conf"
    ),
    "API_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/api.conf"
    ),
    "CACHE_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/cache.conf"
    ),
    "LOGGING_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/logging.conf"
    ),
}
# Enhanced logging configuration for detailed output and debugging
# Define a comprehensive logging configuration dictionary
# Correctly accessing the 'LOGGING_CONF' file path from the FILES dictionary
logging_conf_path: FILE_PATH = FILES["LOGGING_CONF"]

# Ensure the logging configuration file exists before proceeding
logging_setup = asyncio.run(ensure_logging_config_exists(logging_conf_path))

# Since the logging configuration is now guaranteed to exist, we can proceed to use it as intended
logging_config: Dict[str, Any] = DEFAULT_LOGGING_CONFIG
# Apply the logging configuration to initialize the logging system according to the defined specifications.
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


class CythonCompiler:
    """
    Manages the compilation of Python code into Cython C extensions, ensuring that the C version is
    up-to-date with its Python counterpart, and handles dynamic execution based on code integrity.
    This class encapsulates the entire process of converting Python code into a Cython C extension,
    from extracting the source code of the target object, generating a hash for version control,
    to compiling the Cython code and dynamically executing the compiled C extension. It is designed
    to be used both as a standalone compiler instance or as a decorator, providing flexibility in
    its application.
    """

    @classmethod
    @UniversalDecorator()
    async def create(
        cls, obj: Optional[Type] = None
    ) -> Union[Callable[..., Awaitable], "CythonCompiler"]:
        """
        Creates a new instance of the CythonCompiler class asynchronously or acts as a decorator
        if no object is provided directly. This method serves as an entry point for utilizing the
        CythonCompiler, allowing for both direct instantiation with a given Python object or as a
        decorator to dynamically compile and execute the decorated object as a Cython C extension.

        Args:
            obj (Type, optional): The Python object to be compiled into a Cython C extension. Defaults to None.

        Returns:
            Union[Callable[..., Awaitable], CythonCompiler]: A CythonCompiler instance or a decorator function.
        """
        if obj is None:
            # If no object is provided, return a decorator that takes the object to be decorated
            async def decorator(decorated_obj: Type) -> "CythonCompiler":
                source_code: str = await cls._extract_source_code(decorated_obj) or ""
                return cls(decorated_obj, source_code)

            return decorator
        else:
            # If obj is provided, create and return the CythonCompiler instance directly
            source_code: str = await cls._extract_source_code(obj) or ""
            return cls(obj, source_code)

    @UniversalDecorator()
    def __init__(self, obj: Type, source_code: str) -> None:
        """
        Initializes the CythonCompiler with a Python object, typically a function or class, and its
        corresponding source code. This constructor method prepares the compiler instance by setting
        up the necessary attributes for the compilation process, including the target object, its
        source code, and the filenames for the generated Cython and hash files.

        Args:
            obj (Type): The Python object to be compiled into a Cython C extension.
            source_code (str): The source code of the Python object.
        """
        self.obj: Type = obj
        self.module_name: str = f"{obj.__name__}_{self._generate_hash()}"
        self.source_code: str = source_code
        self.pyx_file: str = f"{self.module_name}.pyx"
        self.hash_file: str = f"{self.module_name}.hash"

    @UniversalDecorator()
    async def _extract_source_code(self, obj: Type) -> str:
        """
        Extracts and concatenates the source code of a Python object and all nested members. This method
        recursively traverses the target object, extracting the source code of the object itself and any
        nested functions or classes, concatenating them into a single string. This comprehensive extraction
        process ensures that the compiled Cython C extension is a complete representation of the original
        Python object.

        Args:
            obj (Type): The Python object whose source code is to be extracted.

        Returns:
            str: The concatenated source code of the object and its members.
        """
        source: str = ""
        try:
            source = getsource(obj)
            members = getmembers(obj, predicate=lambda x: isfunction(x) or isclass(x))
            for _, member in members:
                if member is not obj:
                    source += await self._extract_source_code(member)
            return source
        except Exception as e:
            logger.error(f"Failed to extract source code: {e}")
            return source  # Ensures a string is always returned, even in error cases.

    @UniversalDecorator()
    def _generate_hash(self) -> str:
        """
        Generates a SHA-256 hash of the concatenated source code for integrity checking. This hash is used
        to determine if the source code has changed since the last compilation, allowing for efficient
        recompilation only when necessary. The hash generation process is a critical component of the
        CythonCompiler's version control mechanism, ensuring that the compiled C extension is always
        up-to-date with the latest version of the Python source code.

        Returns:
            str: The SHA-256 hash of the source code.
        """
        try:
            return hashlib.sha256(self.source_code.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate hash: {e}")
            return ""

    @UniversalDecorator()
    async def _file_exists_and_matches(
        self, filename: str, expected_content: str
    ) -> bool:
        """
        Checks if a file exists and its content matches the expected content. This method is used to verify
        the integrity of the compiled Cython C extension and its associated hash file, ensuring that the
        compiled code matches the current version of the source code. If the file does not exist or its
        content does not match the expected content, recompilation is triggered.

        Args:
            filename (str): The name of the file to check.
            expected_content (str): The expected content to match against the file's content.

        Returns:
            bool: True if the file exists and its content matches the expected content, False otherwise.
        """
        try:
            if os.path.exists(filename):
                async with aiofiles.open(filename, "r") as file:
                    content = await file.read()
                return content == expected_content
            return False
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return False

    @UniversalDecorator()
    async def _write_to_file(self, filename: str, content: str) -> None:
        """
        Asynchronously writes content to a file. This method is utilized to write the compiled Cython source code
        to a .pyx file and to update the hash file with the latest hash of the source code. The asynchronous file
        writing process ensures that the compilation workflow does not block other operations, maintaining the
        efficiency of the overall system.

        Args:
            filename (str): The name of the file to write to.
            content (str): The content to write to the file.
        """
        try:
            async with aiofiles.open(filename, "w") as file:
                await file.write(content)
        except Exception as e:
            logger.error(f"Failed to write to file {filename}: {e}")

    @UniversalDecorator()
    def _compile_cython(self) -> None:
        """
        This method is responsible for the compilation of Python source code into a Cython C extension.
        It achieves this by writing the source code to a .pyx file and then invoking the Cython compiler
        to generate a C extension in place. This process is crucial for converting Python code into
        a more performant C extension, thereby enhancing execution speed while maintaining the original
        logic and functionality of the code.

        Raises:
            Exception: Captures and logs any exception that occurs during the compilation process,
                       ensuring that the system remains robust and the error is clearly reported.
        """
        try:
            # Open or create the .pyx file and write the source code into it.
            with open(self.pyx_file, "w") as file:
                file.write(self.source_code)
            # Utilize the Cython compiler to generate a C extension from the .pyx file.
            setup(
                ext_modules=cythonize(
                    Extension(self.module_name, sources=[self.pyx_file])
                ),
                script_args=["build_ext", "--inplace"],
            )
        except Exception as e:
            # Log any exceptions that occur during the compilation process.
            logger.error(f"Compilation failed: {e}")
        finally:
            # Reset the command-line arguments to their original state.
            sys.argv = sys.argv[:1]

    @UniversalDecorator()
    def _execute_c(self) -> None:
        """
        Dynamically imports and executes the compiled C extension. This method is a critical component
        of the runtime execution, allowing for the dynamic importation and execution of the C extension
        generated from the Python source code. This approach ensures that the most recent compilation
        is always executed, reflecting any changes made to the source code.

        Raises:
            ImportError: Captures and logs any ImportError that occurs during the dynamic importation
                         process, providing clear feedback on the failure to import the compiled module.
        """
        try:
            # Dynamically import the compiled C extension using its module name.
            __import__(self.module_name)
        except ImportError as e:
            # Log any ImportError that occurs during the importation of the compiled module.
            logger.error(f"Failed to import module {self.module_name}: {e}")

    @UniversalDecorator()
    async def ensure_latest_version_and_execute(self) -> None:
        """
        Ensures that the latest version of the Python object is compiled to a Cython C extension and executes it.
        This method orchestrates the entire process, from verifying the integrity of the compiled extension
        against the current source code to executing the compiled C extension. It plays a pivotal role in
        maintaining the synchronization between the Python source code and its compiled C counterpart,
        ensuring that any changes in the source code are reflected in the executed C extension.

        Raises:
            Exception: Captures and logs any exception that occurs during the file existence check,
                       compilation, or file writing processes, ensuring robust error handling and clear reporting.
        """
        current_hash = self._generate_hash()
        if await self._file_exists_and_matches(self.hash_file, current_hash):
            logger.info(
                f"No changes detected. Executing C version of {self.obj.__name__}."
            )
            self._execute_c()
        else:
            logger.info(
                f"Changes detected or C version not found. Updating {self.obj.__name__}."
            )
            self._compile_cython()
            await self._write_to_file(self.hash_file, current_hash)
            self._execute_c()

    from inspect import iscoroutinefunction


@UniversalDecorator()
def cythonize_function(obj: Type) -> Callable[..., Awaitable]:
    async def async_wrapper(*args, **kwargs):
        compiler = await CythonCompiler.create(obj)
        await compiler.ensure_latest_version_and_execute()
        if asyncio.iscoroutinefunction(obj):
            return await obj(*args, **kwargs)
        else:
            return obj(*args, **kwargs)

    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_wrapper(*args, **kwargs))

    if asyncio.iscoroutinefunction(obj):
        return async_wrapper
    else:
        return sync_wrapper


# Example function decorated to demonstrate the functionality
@cythonize_function
def example_function()
    


"""
TODO List for Further Refinements:
- Maximize multiprocessing and asynchronous methods in all possible aspects for parallelized operation of maximum efficacy and efficiency.
- Optimize Source Code Extraction: Enhance the recursive extraction logic to handle more complex structures and edge cases.
- Improve Error Handling: Expand error handling to cover more specific exceptions, particularly during file operations and dynamic imports.
- Implement a caching mechanism for compiled C extensions to avoid recompilation of unchanged code, enhancing efficiency.
- Explore the integration of more advanced Cython features to further optimize the compiled C extensions.
- Develop a more robust system for managing and cleaning up generated files (.pyx, .c, .so) to prevent clutter.
- Investigate the possibility of integrating with other Python-to-C compilers or tools for broader compatibility and optimization options.
"""
