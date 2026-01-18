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
class ColorManager:
    """
    A class meticulously designed to manage the color cycling of models within a 3D environment, ensuring dynamic visual effects.
    """

    def __init__(self, render_node, initial_hue_angle=0.0):
        """
        Initialize the ModelColorManager with a specific render node and an initial hue angle for color management.

        Parameters:
        - render_node (NodePath): The render node to which the color changes will be applied.
        - initial_hue_angle (float): The initial hue angle in degrees, used to set the starting color.
        """
        self.render_node = render_node
        self.hue_angle = np.array([initial_hue_angle], dtype=np.float64)

    def cycle_model_colors(self, task):
        """
        Methodically update the color of the models based on a hue rotation, ensuring a continuous and visually appealing change.

        Parameters:
        - task (Task): A task object that provides context, particularly the elapsed time since the task began.

        Returns:
        - Task.cont: A constant indicating that the task should continue running.
        """
        self.hue_angle = (self.hue_angle + 0.5) % 360
        logging.debug('Updated hue angle for color cycling: {}'.format(self.hue_angle))
        rgba_color = convert_hpr_to_vbase4_with_full_opacity(self.hue_angle[0], 0, 0)
        logging.debug('Converted RGBA color: {}'.format(rgba_color))
        geom_nodes = self.render_node.findAllMatches('**/+GeomNode')
        for index, node in enumerate(geom_nodes):
            node.node().setAttrib(ColorAttrib.makeFlat(rgba_color))
            logging.info('Applied new color to node {}: {}'.format(index, node))
        return Task.cont