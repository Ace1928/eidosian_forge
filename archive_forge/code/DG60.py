import pyopencl as cl
import numpy as np
import functools
import os
import logging
from typing import Any, Dict, Tuple
from collections import deque
import pickle


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
        self.shader_cache = functools.lru_cache(
            maxsize=128
        )  # Using LRU cache to store up to 128 compiled shaders
        self.pinned_memory_buffers = {}

    def compile_shader(self, source: str, shader_type: str) -> cl.Program:
        """
        Compiles a shader from source code based on the specified shader type, utilizing advanced error handling and logging for robustness.

        Parameters:
            source (str): The source code of the shader.
            shader_type (str): The type of shader to compile. Must be one of 'vertex', 'fragment', 'geometry', 'tess_control', 'tess_evaluation', 'compute'.

        Returns:
            cl.Program: The compiled shader program.

        Raises:
            ValueError: If an unsupported shader type is provided.
        """
        supported_shaders = [
            "vertex",
            "fragment",
            "geometry",
            "tess_control",
            "tess_evaluation",
            "compute",
        ]
        if shader_type not in supported_shaders:
            logging.error(f"Unsupported shader type provided: {shader_type}")
            raise ValueError("Unsupported shader type provided.")

        # Check if the shader is already compiled and cached
        cache_key = (source, shader_type)
        if cache_key in self.shader_cache:
            logging.info(f"Shader retrieved from cache: {shader_type}")
            return self.shader_cache[cache_key]

        try:
            # Create and build the program
            program = cl.Program(self.context, source).build()
            self.shader_cache[cache_key] = program
            logging.info(f"Shader compiled and cached: {shader_type}")
            return program
        except cl.ProgramBuildError as e:
            logging.error(f"Failed to compile shader: {e}")
            raise

    def load_shader_from_file(self, file_path: str, shader_type: str) -> cl.Program:
        """
        Loads and compiles a shader from a file based on the specified shader type, employing comprehensive error handling and data management.

        Parameters:
            file_path (str): The path to the file containing the shader source code.
            shader_type (str): The type of shader to compile. Must be one of 'vertex', 'fragment', 'geometry', 'tess_control', 'tess_evaluation', 'compute'.

        Returns:
            cl.Program: The compiled shader program.

        Raises:
            FileNotFoundError: If the shader file does not exist.
            ValueError: If an unsupported shader type is provided.
        """
        if not os.path.exists(file_path):
            logging.error(f"Shader file not found: {file_path}")
            raise FileNotFoundError(f"Shader file not found: {file_path}")

        with open(file_path, "r") as file:
            shader_source = file.read()

        return self.compile_shader(shader_source, shader_type)
