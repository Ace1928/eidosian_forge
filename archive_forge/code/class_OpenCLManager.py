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
class OpenCLManager:
    """
    Manages OpenCL operations to leverage parallel computing capabilities of GPUs, optimizing computational tasks that can be performed concurrently.
    """

    def __init__(self, gpu_manager):
        """
        Initializes the OpenCLManager with a reference to an instance of GPUManager to facilitate access to the GPU context and command queue.

        Parameters:
            gpu_manager (GPUManager): An instance of GPUManager which provides the necessary GPU context and command queue for OpenCL operations.
        """
        self.gpu_manager = gpu_manager
        logging.info('OpenCLManager initialized with GPUManager.')

    @lru_cache(maxsize=128)
    def create_program(self, source_code: str) -> cl.Program:
        """
        Compiles OpenCL source code into a program using the GPU context provided by the GPUManager.

        Parameters:
            source_code (str): The OpenCL source code as a string.

        Returns:
            cl.Program: The compiled OpenCL program.

        Raises:
            cl.ProgramBuildError: If there is an error during the building of the OpenCL program.
        """
        try:
            program = cl.Program(self.gpu_manager.context, source_code).build()
            logging.info('OpenCL program created and built from source.')
            return program
        except cl.ProgramBuildError as e:
            logging.error(f'Failed to build OpenCL program: {e}')
            raise

    def execute_program(self, program: cl.Program, data: np.ndarray):
        """
        Executes an OpenCL program with provided data, handling data transfer and kernel execution.

        Parameters:
            program (cl.Program): The OpenCL program to execute.
            data (np.ndarray): The data to process, expected as a NumPy array for efficient handling.

        Raises:
            cl.LogicError: If there is an error during the execution of the program.
        """
        try:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                logging.debug('Data converted to NumPy array for efficient processing.')
            mem_flags = cl.mem_flags
            data_buffer = cl.Buffer(self.gpu_manager.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=data)
            logging.debug('Data buffer created for OpenCL program execution.')
            kernel = cl.Kernel(program, 'process_data')
            kernel.set_arg(0, data_buffer)
            cl.enqueue_nd_range_kernel(self.gpu_manager.queue, kernel, data.shape, None)
            self.gpu_manager.queue.finish()
            logging.info(f'Executed OpenCL program with data: {data}')
        except cl.LogicError as e:
            logging.error(f'Error executing OpenCL program: {e}')
            raise