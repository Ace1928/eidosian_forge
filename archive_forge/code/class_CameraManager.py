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
class CameraManager:
    """
    Manages cameras within the environment, controlling their positioning, orientation, and parameters to capture and display the scene effectively.
    This class utilizes an OrderedDict to maintain the insertion order of cameras, which can be beneficial for certain rendering optimizations.
    """

    def __init__(self):
        """
        Initializes the CameraManager with an empty ordered dictionary to store camera data and sets the active camera to None.
        """
        self.cameras: Dict[str, np.ndarray] = OrderedDict()
        self.active_camera: Optional[str] = None
        logging.info('CameraManager initialized with no cameras and no active camera.')

    def add_camera(self, camera_id: str, camera_data: np.ndarray) -> None:
        """
        Adds a camera to the system, specifying its setup and operational parameters, stored as a NumPy structured array for efficient data handling.

        Parameters:
            camera_id (str): The unique identifier for the camera.
            camera_data (np.ndarray): The structured array containing camera parameters such as position, orientation, and field of view.
        """
        self.cameras[camera_id] = camera_data
        logging.debug(f'Camera added: {camera_id} with data {camera_data}')

    def select_camera(self, camera_id: str) -> None:
        """
        Selects a camera as the active camera, directing the rendering process to use its view. Logs the action and handles the case where the camera ID is not found.

        Parameters:
            camera_id (str): The unique identifier for the camera to be activated.
        """
        if camera_id in self.cameras:
            self.active_camera = camera_id
            logging.info(f'Active camera set to: {camera_id}')
        else:
            logging.error(f'Camera ID {camera_id} not found.')
            raise ValueError(f'Camera ID {camera_id} not found.')