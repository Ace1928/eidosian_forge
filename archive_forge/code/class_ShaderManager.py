import pyopencl as cl  # https://documen.tician.de/pyopencl/ - Used for managing and executing OpenCL commands on GPUs.
import OpenGL.GL as gl  # https://pyopengl.sourceforge.io/documentation/ - Used for executing OpenGL commands for rendering graphics.
import json  # https://docs.python.org/3/library/json.html - Used for parsing and outputting JSON formatted data.
import numpy as np  # https://numpy.org/doc/ - Used for numerical operations on arrays and matrices.
import functools  # https://docs.python.org/3/library/functools.html - Provides higher-order functions and operations on callable objects.
import logging  # https://docs.python.org/3/library/logging.html - Used for logging events and messages during execution.
from pyopencl import (
import hashlib  # https://docs.python.org/3/library/hashlib.html - Used for hashing algorithms.
import pickle  # https://docs.python.org/3/library/pickle.html - Used for serializing and deserializing Python objects.
from typing import (
from functools import (
class ShaderManager:
    """
    Manages shader programs that are used to control the rendering pipeline. This includes compiling, loading, and maintaining vertex and fragment shaders.
    """

    def __init__(self):
        """
        Initializes the ShaderManager with an empty dictionary to store shader programs.
        """
        self.shaders: Dict[int, Tuple[str, str]] = {}
        logging.info('ShaderManager initialized with an empty shader storage.')

    def compile_shader(self, source_code: str, shader_type: str) -> None:
        """
        Compiles a shader from source code and logs the operation.

        Parameters:
        source_code (str): The GLSL source code of the shader.
        shader_type (str): The type of the shader (e.g., 'vertex', 'fragment').

        Returns:
        None
        """
        shader_id = self._generate_shader_id(source_code, shader_type)
        self.shaders[shader_id] = (source_code, shader_type)
        logging.debug(f'Shader compiled and stored: ID={shader_id}, Type={shader_type}')

    def get_shader(self, shader_id: int) -> Optional[Tuple[str, str]]:
        """
        Retrieves a compiled shader by its identifier.

        Parameters:
        shader_id (int): The unique identifier of the shader.

        Returns:
        Optional[Tuple[str, str]]: The shader source code and type if found, None otherwise.
        """
        shader = self.shaders.get(shader_id, None)
        if shader is None:
            logging.error(f'Shader ID {shader_id} not found.')
        else:
            logging.debug(f'Shader retrieved: ID={shader_id}, Type={shader[1]}')
        return shader

    def _generate_shader_id(self, source_code: str, shader_type: str) -> int:
        """
        Generates a unique identifier for a shader using its source code and type.

        Parameters:
        source_code (str): The GLSL source code of the shader.
        shader_type (str): The type of the shader.

        Returns:
        int: A unique identifier for the shader.
        """
        hasher = hashlib.sha256()
        hasher.update(source_code.encode('utf-8'))
        hasher.update(shader_type.encode('utf-8'))
        shader_id = int(hasher.hexdigest(), 16) % 10 ** 8
        logging.debug(f'Generated shader ID {shader_id} for type {shader_type}')
        return shader_id