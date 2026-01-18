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
DIR_NAME: TypeAlias = str
DIR_PATH: TypeAlias = pathlib.Path
FILE_NAME: TypeAlias = str
FILE_PATH: TypeAlias = pathlib.Path

# Immutable Codebase-Wide Constants
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


class CythonCompiler:
    def __init__(self, obj: Type, module_name: str, hash_file: FILE_PATH):
        self.obj = obj
        self.module_name = module_name
        self.hash_file = hash_file
        self.logger = logging.getLogger(__name__)
        ensure_logging_config_exists(LOGGING_CONF=FILES["LOGGING_CONF"])
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

    @staticmethod
    def _generate_hash() -> str:
        """
        Generates a unique hash for the current state of the source code of the object.
        This hash is used to determine if the source code has changed since the last compilation.
        """
        source = getsource(self.obj)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    async def _file_exists_and_matches(
        self, file_path: FILE_PATH, current_hash: str
    ) -> bool:
        """
        Checks if the hash file exists and if its content matches the current hash.
        """
        if not file_path.exists():
            return False
        async with aiofiles.open(file_path, mode="r") as file:
            stored_hash = await file.read()
        return stored_hash == current_hash

    async def _write_to_file(self, file_path: FILE_PATH, content: str) -> None:
        """
        Writes the given content to the specified file.
        """
        async with aiofiles.open(file_path, mode="w") as file:
            await file.write(content)

    def _compile_cython(self) -> None:
        """
        Compiles the Python source code to a Cython C extension.
        """
        setup(
            ext_modules=cythonize(
                Extension(
                    name=self.module_name,
                    sources=[getsource(self.obj)],
                )
            )
        )

    def _execute_c(self) -> None:
        """
        Dynamically imports and executes the compiled C extension.
        """
        try:
            __import__(self.module_name)
        except ImportError as e:
            self.logger.error(f"Failed to import module {self.module_name}: {e}")

    @UniversalDecorator()
    async def ensure_latest_version_and_execute(self) -> None:
        """
        Ensures the latest version of the Python object is compiled to a Cython C extension and executes it.
        """
        current_hash = self._generate_hash()
        if await self._file_exists_and_matches(self.hash_file, current_hash):
            self.logger.info(
                f"No changes detected. Executing C version of {self.obj.__name__}."
            )
            self._execute_c()
        else:
            self.logger.info(
                f"Changes detected or C version not found. Updating {self.obj.__name__}."
            )
            self._compile_cython()
            await self._write_to_file(self.hash_file, current_hash)
            self._execute_c()


@UniversalDecorator()
def cythonize_function(obj: Type) -> Callable[..., Awaitable]:
    async def async_wrapper(*args, **kwargs):
        compiler = CythonCompiler(
            obj, module_name=obj.__name__, hash_file=FILES["HASH"]
        )
        await compiler.ensure_latest_version_and_execute()
        if iscoroutinefunction(obj):
            return await obj(*args, **kwargs)
        else:
            return obj(*args, **kwargs)

    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_wrapper(*args, **kwargs))

    if iscoroutinefunction(obj):
        return async_wrapper
    else:
        return sync_wrapper


import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, AsyncGenerator, Callable, Awaitable, Type
from inspect import iscoroutinefunction
from functools import wraps


class FractalAlgorithmsLibrary:
    """
    This library provides various algorithms for generating fractal data.
    The libraries it contains are:
    - Mandelbulb
    - Mandelbox
    - Julibrot
    - Quaternion Julia
    - Hypercomplex Fractals
    - Sierpinski Tetrahedron
    - Menger Sponge
    - Koch Snowflake 3D
    - IFS
    - Fractal Flames
    - Apollonian Gasket
    - Circle Inversion
    - Fractal Terrains
    - Strange Attractors

    Each algorithm is implemented as a coroutine function that generates fractal data.
    These are then fed into the generate fractal data to create the final fractal and determine the number of sliders to be produced for the gui.
    So that the user can control the fractals. The fractal type being selected from a dropdown box and determining the variables and sliders.
    The sliders are then used to control the fractal and the generate fractal data function is called with the sliders as the parameters.

    The fractal data is then generated in chunks and displayed in real-time using matplotlib. Updating continuously and managing memory to prevent overflow.
    Allowing it to run indefinitely and be stopped at any time. The fractal data is then saved to a file for later use and analysis.

    Methods:
        - mandelbulb: Generates 3D fractal data for the Mandelbulb set.
        - mandelbox: Generates 3D fractal data for the Mandelbox set.
        - julibrot: Generates 2D fractal data for the Julibrot set.
        - quaternion_julia: Generates 3D fractal data for the Quaternion Julia set.
        - hypercomplex_fractals: Generates 3D fractal data for hypercomplex fractals.
        - sierpinski_tetrahedron: Generates 3D fractal data for the Sierpinski Tetrahedron.
        - menger_sponge: Generates 3D fractal data for the Menger Sponge.
        - koch_snowflake_3d: Generates 3D fractal data for the Koch Snowflake.
        - ifs: Generates 2D fractal data for Iterated Function Systems.
        - fractal_flames: Generates 2D fractal data for Fractal Flames.
        - apollonian_gasket: Generates 2D fractal data for the Apollonian Gasket.
        - circle_inversion: Generates 2D fractal data for Circle Inversion.
        - fractal_terrains: Generates 3D fractal data for Fractal Terrains.
        - strange_attractors: Generates 3D fractal data for Strange Attractors.

    Attributes:
        - fractal_types: A list of supported fractal types.
        - fractal_functions: A dictionary mapping fractal types to their corresponding coroutine functions.
        - fractal_sliders: A dictionary mapping fractal types to the number of sliders required for the gui.
        - fractal_data: A dictionary mapping fractal types to the generated fractal data.
        - fractal_plot: A dictionary mapping fractal types to the matplotlib plot of the generated fractal data.
        - fractal_animation: A dictionary mapping fractal types to the matplotlib animation of the generated fractal data.
        - fractal_file: A dictionary mapping fractal types to the file path where the generated fractal data is saved.
        - fractal_data_chunks: A dictionary mapping fractal types to the chunks of fractal data generated.
        - fractal_data_generator: A dictionary mapping fractal types to the asynchronous generator function for generating fractal data.
        - fractal_plot_generator: A dictionary mapping fractal types to the asynchronous generator function for plotting the fractal data.
        - fractal_save_generator: A dictionary mapping fractal types to the asynchronous generator function for saving the fractal data.
        - fractal_stop: A dictionary mapping fractal types to a boolean value indicating whether the fractal generation should stop.
        - fractal_stop_button: A dictionary mapping fractal types to the stop button for the gui.
        - fractal_sliders: A dictionary mapping fractal types to the sliders for the gui.
        - fractal_dropdown: A dictionary mapping fractal types to the dropdown box for selecting the fractal type in the gui.
        - fractal_gui: A dictionary mapping fractal types to the gui for controlling the fractal.
        - plot_fractal_async: A coroutine function for plotting a fractal asynchronously.
        - generate_fractal_data: An asynchronous generator function for generating fractal data in chunks.
        - save_fractal: A coroutine function for saving the generated fractal data to a file.
        - stop_fractal: A coroutine function for stopping the generation of fractal data.
        - update_fractal: A coroutine function for updating the generated fractal data.
        - reset_fractal: A coroutine function for resetting the generated fractal data.
        - start_fractal: A coroutine function for starting the generation of fractal data.
        - pause_fractal: A coroutine function for pausing the generation of fractal data.
        - resume_fractal: A coroutine function for resuming the generation of fractal data.
        - graceful_shutdown: A coroutine function for gracefully shutting down the fractal generation process.
        - main: The main entry point for running the fractal generation process.

    Example Usage:
        - Initialize the FractalAlgorithmsLibrary object.
        - Select a fractal type from the dropdown box.
        - Adjust the sliders to control the fractal.
        - Start the fractal generation process.
        - Pause, resume, or stop the fractal generation as needed.
        - Save the generated fractal data to a file for later analysis.
        - Gracefully shut down the fractal generation process when done.
    """


async def generate_fractal_data(
    steps: int, zoom: float
) -> AsyncGenerator[np.ndarray, None]:
    """
    Asynchronously generates fractal data in chunks based on the given number of steps and zoom level.

    This function generates 3D fractal data for visualization. It uses asynchronous programming innovatively
    to generate data in chunks representing the appropriate level of detail for the fractal. The data is
    yielded as numpy arrays, allowing for real-time updates of the fractal visualization as more data is generated.
    The generated fractal can take a variety of parameters to alter how it looks, a total of # of paremeters composed of:

    Args:
        steps (int): The number of steps to generate the fractal.
        zoom (float): The zoom level for fractal detail.

    Returns:
        - AsyncGenerator[np.ndarray, None] : A chunk of the fractal data as a numpy array.
    """


async def plot_fractal_async(steps: int = 1000, zoom: float = 1.0) -> None:
    """
    Asynchronously plots a fractal, updating the plot with more detail as more data is generated.

    This function demonstrates the use of asynchronous programming in data visualization. It leverages
    the asynchronous generation of fractal data and updates a matplotlib plot with each new chunk of data
    received. This approach allows for the plot to be dynamically updated in real-time as the data is generated.

    Args:
        steps (int, optional): The number of steps to generate the fractal. Defaults to 1000.
        zoom (float, optional): The zoom level for fractal detail. Defaults to 1.0.

    The function initializes a matplotlib figure and axis, then asynchronously fetches chunks of fractal data,
    updating the plot with each new chunk. This is achieved through the use of an asynchronous for loop and the
    FuncAnimation class from matplotlib, which is used to animate the plot updates. The plot is displayed using
    plt.show(), which blocks execution until the plot window is closed.
    """
    fig, ax = plt.subplots()
    fractal_data = np.zeros((10, 10))

    def update_plot(data: np.ndarray) -> None:
        """
        Updates the plot with new fractal data.

        This function is called with each new chunk of fractal data generated by the asynchronous generator.
        It updates the fractal_data array with the new data and then redraws the plot with the updated data.

        Args:
            data (np.ndarray): The latest chunk of fractal data.
        """
        nonlocal fractal_data
        fractal_data += data  # Update fractal data with new chunk
        ax.clear()
        ax.imshow(fractal_data)

    async def fetch_and_draw() -> None:
        """
        Fetches chunks of fractal data asynchronously and updates the plot for each chunk.

        This coroutine function is responsible for fetching the fractal data asynchronously from the generator
        and then calling update_plot with each new chunk of data. It yields control back to the event loop between
        each update to ensure smooth animation and responsiveness of the plot.
        """
        async for data_chunk in generate_fractal_data(steps, zoom):
            update_plot(data_chunk)
            plt.draw()
            await asyncio.sleep(
                0.01
            )  # Give control back to the event loop for smooth animation

    # The FuncAnimation requires a callable that returns an iterable of artists; however, fetch_and_draw is a coroutine.
    # To integrate it with FuncAnimation, we adapt it by creating a wrapper function that schedules the coroutine
    # and ensures it is run in the event loop, effectively bridging the gap between synchronous and asynchronous code.
    def animation_frame_generator() -> Callable[..., Awaitable[None]]:
        """
        Wraps the asynchronous fetch_and_draw coroutine to make it compatible with matplotlib.animation.FuncAnimation.

        This function returns a callable that, when called, schedules the fetch_and_draw coroutine in the asyncio event loop.
        It is designed to bridge the gap between synchronous code expected by FuncAnimation and the asynchronous nature of
        fetch_and_draw, allowing for the dynamic updating of the plot with asynchronously generated fractal data.

        Returns:
            Callable[..., Awaitable[None]]: A callable that schedules the fetch_and_draw coroutine.
        """
        return asyncio.create_task(fetch_and_draw())

    ani = FuncAnimation(
        fig, lambda x: None, frames=animation_frame_generator, repeat=False
    )
    plt.show()


# Example usage
# asyncio.run(plot_fractal_async(1000, 2.0))


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
