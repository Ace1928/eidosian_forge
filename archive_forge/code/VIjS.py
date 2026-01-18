import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task

# Configure logging with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_hue_to_rgb_vector(hue_angle_degrees):
    """
    Convert a hue value (in degrees) to an RGB color vector with full saturation and brightness.
    This function meticulously calculates the RGB values based on the hue angle provided, ensuring
    the output is a tuple of RGB values, each component ranging from 0 to 1.

    Parameters:
    - hue_angle_degrees (float): The hue angle in degrees, which will be normalized to a range of 0-360.

    Returns:
    - tuple: A tuple representing the RGB color (r, g, b), each component as a float from 0 to 1.
    """
    # Normalize the hue angle to ensure it is within the range of 0 to 360 degrees
    normalized_hue_angle = hue_angle_degrees % 360

    # Define constants for maximum saturation and brightness
    maximum_saturation = 1.0
    maximum_brightness = 1.0

    # Calculate the intermediate value 'x' used in the RGB conversion process
    intermediate_x = 1 - abs((normalized_hue_angle / 60.0 % 2) - 1)

    # Define the base adjustment for RGB values, which is zero in this case as no adjustment is needed
    base_rgb_adjustment = 0.0

    # Define the RGB sectors based on the hue angle using a structured array approach
    rgb_sector_matrix = np.array(
        [
            (maximum_saturation, intermediate_x, 0),
            (intermediate_x, maximum_saturation, 0),
            (0, maximum_saturation, intermediate_x),
            (0, intermediate_x, maximum_saturation),
            (intermediate_x, 0, maximum_saturation),
            (maximum_saturation, 0, intermediate_x),
        ],
        dtype=np.float32,
    )

    # Determine the sector index based on the normalized hue angle
    sector_index = int(normalized_hue_angle // 60)

    # Extract the RGB values from the sector matrix based on the calculated sector index
    rgb_values = rgb_sector_matrix[sector_index]

    # Adjust the RGB values by adding the base RGB adjustment
    adjusted_rgb_values = np.array(rgb_values) + base_rgb_adjustment

    # Return the RGB values as a tuple
    return tuple(adjusted_rgb_values)


def from_hpr_to_vbase4(hue, pitch, roll):
    """
    Convert HPR (Hue, Pitch, Roll) to a VBase4 object, using hue for color.
    """
    r, g, b = hue_to_rgb(hue)
    alpha = 1.0  # Full opacity
    return VBase4(r, g, b, alpha)


class ModelShowcase(ShowBase):
    """
    A class to render 3D models from models.py with dynamic lighting, color cycling, and camera manipulation.
    """

    def __init__(self):
        """
        Initialize the ModelShowcase with enhanced rendering settings and tasks.
        """
        super().__init__()
        self.model_index = 0
        self.model_names = self.retrieve_model_names()

    from models import (
        construct_triangle_sheet_with_vertex_data,
        construct_square_sheet_with_vertex_data,
        construct_circle_sheet_with_vertex_data,
        construct_cube,
        construct_sphere,
        construct_cylinder,
        construct_cone,
        construct_dodecahedron,
        construct_icosahedron,
        construct_octahedron,
        construct_tetrahedron,
        construct_conical_frustum,
        construct_cylindrical_frustum,
        construct_spherical_frustum,
        construct_torus_knot,
        construct_trefoil_knot,
        construct_mobius_strip,
        construct_klein_bottle,
        construct_torus,
    )

    def setup_lights(self):
        """
        Set up various lights in the scene.
        """
        # Point light
        pl = PointLight("pl")
        pl.setColor((1, 1, 1, 1))
        plnp = self.render.attachNewNode(pl)
        plnp.setPos(10, 20, 0)
        self.render.setLight(plnp)

        # Ambient light
        al = AmbientLight("al")
        al.setColor((0.5, 0.5, 0.5, 1))
        alnp = self.render.attachNewNode(al)
        self.render.setLight(alnp)

        # Directional light
        dl = DirectionalLight("dl")
        dl.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dl)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

    def spin_camera_task(self, task):
        """
        Task to spin the camera around the scene.
        """
        angle_degrees = task.time * 6.0
        angle_radians = angle_degrees * (math.pi / 180.0)
        self.camera.setPos(20 * np.sin(angle_radians), -20 * np.cos(angle_radians), 3)
        self.camera.lookAt(0, 0, 0)
        return Task.cont

    def update_color_task(self, task):
        """
        Task to dynamically update the color of the models.
        """
        self.hue = (self.hue + 0.5) % 360
        color = from_hpr_to_vbase4(self.hue, 0, 0)
        for node in self.render.findAllMatches("**/+GeomNode"):
            node.node().setAttrib(ColorAttrib.makeFlat(color))
        return Task.cont

    def play(self):
        """
        Play the 3D model showcase with dynamic lighting and color cycling.
        """
        self.hue = 0
        self.setup_lights()
        self.taskMgr.add(self.spin_camera_task, "SpinCameraTask")
        self.taskMgr.add(self.update_color_task, "UpdateColorTask")
        self.run()

    def pause(self):
        """
        Pause the 3D model showcase.
        """
        self.taskMgr.remove("SpinCameraTask")
        self.taskMgr.remove("UpdateColorTask")

    def restart(self):
        """
        Restart the 3D model showcase.
        """
        self.pause()
        self.play()

    def next_model(self):
        """
        Load the next 3D model in the showcase.
        """
        self.model_index = (self.model_index + 1) % len(self.model_names)
        self.load_model(self.model_names[self.model_index])

    def previous_model(self):
        """
        Load the previous 3D model in the showcase.
        """
        self.model_index = (self.model_index - 1) % len(self.model_names)
        self.load_model(self.model_names[self.model_index])

    def load_model(self, model_name):
        """
        Load a specific 3D model in the showcase.
        """
        model = getattr(self, f"construct_{model_name}")()
        model.reparentTo(self.render)

    def clear_scene(self):
        """
        Clear the scene of all 3D models.
        """
        self.render.removeNode()

    def retrieve_model_names(self):
        """
        Retrieve the available 3D models in the showcase.
        """
        model_names = [name[10:] for name in dir(self) if name.startswith("construct_")]
        return model_names

    def run(self):
        """
        Run the Panda3D application.
        """
        self.taskMgr.run()


def main():
    """
    Main function to run the 3D model showcase.
    """
    showcase = ModelShowcase()
    showcase.play()


if __name__ == "__main__":
    main()
