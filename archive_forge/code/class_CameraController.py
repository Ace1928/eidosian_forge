import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task
from models import (
class CameraController:
    """
    A class meticulously designed to manage the orientation and position of a camera within a 3D environment, ensuring dynamic interaction with the scene.
    """

    def __init__(self, camera):
        """
        Construct the CameraController with a specific camera object, ensuring that the camera is ready for operations like spinning around a scene.

        Parameters:
        - camera (Camera): The camera object that this controller will manage.
        """
        self.camera = camera
        logging.debug('CameraController initialized with camera: {}'.format(camera))

    def spin_camera(self, task):
        """
        Methodically rotate the camera around the scene based on the elapsed time, ensuring a continuous and smooth motion.

        Parameters:
        - task (Task): A task object that provides context, particularly the elapsed time since the task began.

        Returns:
        - Task.cont: A constant indicating that the task should continue running.
        """
        angle_degrees = task.time * 6.0
        logging.debug('Calculated angle in degrees: {}'.format(angle_degrees))
        angle_radians = angle_degrees * (np.pi / 180.0)
        logging.debug('Converted angle in radians: {}'.format(angle_radians))
        position_vector = np.array([20 * np.sin(angle_radians), -20 * np.cos(angle_radians), 3])
        logging.debug('Calculated position vector: {}'.format(position_vector))
        self.camera.setPos(tuple(position_vector))
        logging.info('Camera position set to: {}'.format(tuple(position_vector)))
        self.camera.lookAt((0, 0, 0))
        logging.info('Camera oriented to look at the origin.')
        return Task.cont