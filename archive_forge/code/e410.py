import pyopencl as cl
import numpy as np
import functools
import os
import logging
from typing import Any, Dict, Tuple
from collections import deque
import pickle
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ShaderManager:
    """
    A class meticulously designed to manage shader operations within a GPU context, specifically tailored to compile and manage shaders
    for rendering in a graphics and physics engine. This manager supports a comprehensive range of shaders including vertex, fragment, geometry,
    tessellation control, tessellation evaluation, and compute shaders.

    Attributes:
        context (cl.Context): The OpenCL context associated with a specific device where shaders will be compiled and managed.
        shader_cache (Dict[Tuple[str, str], cl.Program]): Cache to store compiled shader programs to avoid recompilation, utilizing an LRU cache mechanism.
        pinned_memory_buffers (Dict[str, cl.Buffer]): A dictionary to manage pinned memory buffers for efficient host-to-device data transfer.
    """

    def __init__(self, context: cl.Context):
        """
        Initializes the ShaderManager with a given OpenCL context, setting up an LRU cache for compiled shaders and preparing for pinned memory usage.

        Parameters:
            context (cl.Context): The OpenCL context to be used for shader operations.
        """
        self.context = context
        self.shader_cache = self.get_shader_program  # Refer to the caching method
        self.pinned_memory_buffers = {}

    @functools.lru_cache(maxsize=128)
    def get_shader_program(self, source: str, shader_type: str) -> cl.Program:
        """
        Retrieves a shader program from the cache or compiles it if not present.

        Parameters:
            source (str): The source code of the shader.
            shader_type (str): The type of shader to compile.

        Returns:
            cl.Program: The compiled or cached shader program.
        """
        cache_key = (source, shader_type)
        if cache_key in self.shader_cache:
            logging.info(f"Shader retrieved from cache: {shader_type}")
            return self.shader_cache[cache_key]

        try:
            program = cl.Program(self.context, source).build()
            self.shader_cache[cache_key] = program
            logging.info(f"Shader compiled and cached: {shader_type}")
            return program
        except cl.ProgramBuildError as e:
            logging.error(f"Failed to compile shader: {e}")
            raise

    def compile_shader(self, source: str, shader_type: str) -> cl.Program:
        """
        Compiles a shader from source code based on the specified shader type.

        Parameters:
            source (str): The source code of the shader.
            shader_type (str): The type of shader to compile.

        Returns:
            cl.Program: The compiled shader program.
        """
        return self.get_shader_program(source, shader_type)

    def load_shader_from_file(self, file_path: str, shader_type: str) -> cl.Program:

        default_shaders = {
            "compute": "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders/compute_shader.glsl",
            "fragment": "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders/fragment_shader.glsl",
            "geometry": "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders/geometry_shader.glsl",
            "tess_control": "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders/tcs_shader.glsl",
            "tess_evaluation": "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders/tes_shader.glsl",
            "vertex": "/media/lloyd/Aurora_M2/extra-repos/pandas3D/shaders/vertex_shader.glsl",
        }

        if not os.path.exists(file_path):
            logging.error(f"Shader file not found: {file_path}")
            if shader_type in default_shaders and os.path.exists(
                default_shaders[shader_type]
            ):
                logging.info(f"Using default shader for {shader_type}")
                file_path = default_shaders[shader_type]
                # Copy default shader to application's default location
                default_destination = os.path.join(
                    self.context.default_shader_directory, os.path.basename(file_path)
                )
                shutil.copyfile(file_path, default_destination)
                logging.info(
                    f"Copied default shader from {file_path} to {default_destination}"
                )
                file_path = default_destination
            else:
                error_message = f"No default shader available for {shader_type}"
                logging.error(error_message)
                raise FileNotFoundError(error_message)

        with open(file_path, "r") as file:
            shader_source = file.read()

        return self.compile_shader(shader_source, shader_type)
