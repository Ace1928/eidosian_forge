import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task
import numpy as np
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
    # Ensuring that the extraction is done correctly by converting the sector index to an integer if not already
    sector_index_int = int(sector_index)
    # Utilizing advanced numpy techniques to ensure the correct data type and handling
    rgb_values = np.take(rgb_sector_matrix, sector_index_int, axis=0)

    # Adjust the RGB values by adding the base RGB adjustment
    # Using numpy's broadcasting feature to add the base adjustment to each element in the RGB values array
    adjusted_rgb_values = np.add(rgb_values, base_rgb_adjustment)
    # Return the RGB values as a tuple
    return tuple(adjusted_rgb_values)


def convert_hpr_to_vbase4_with_full_opacity(hue, pitch, roll):
    """
    Convert Hue, Pitch, and Roll (HPR) values to a VBase4 object, utilizing the hue component to determine the color.
    This function meticulously transforms the hue component into an RGB color vector, then combines it with a full opacity
    value to create a VBase4 object, which is used extensively in 3D graphics for representing color and transparency.

    Parameters:
    - hue (float): The hue component used for color conversion, representing the angle in degrees on a color wheel.
    - pitch (float): The pitch component, which is not utilized in this function but is included for complete HPR representation.
    - roll (float): The roll component, which is not utilized in this function but is included for complete HPR representation.

    Returns:
    - VBase4: An object representing the RGBA color, where RGB is derived from the hue and A (alpha) is set to 1.0 for full opacity.
    """
    # Utilize the previously defined function to convert hue to an RGB color vector
    rgb_color_vector = convert_hue_to_rgb_vector(hue)

    # Define the alpha value for full opacity
    alpha_value = 1.0

    # Create a VBase4 object with the RGB color vector and full opacity
    color_with_opacity = VBase4(*rgb_color_vector, alpha_value)

    return color_with_opacity


class LightingComponent:
    """
    A class dedicated to managing lighting components within a 3D scene.
    """

    def __init__(
        self,
        light_type: str,
        color: tuple,
        position: tuple = None,
        orientation: tuple = None,
    ):
        """
        Initialize a lighting component with specified type, color, position, and orientation.
        """
        self.light_type = light_type
        self.color = color
        self.position = position
        self.orientation = orientation
        self.light = None
        self.node_path = None

    def create_light(self):
        """
        Create the light based on its type and set its color.
        """
        if self.light_type == "point":
            self.light = PointLight("point_light")
        elif self.light_type == "ambient":
            self.light = AmbientLight("ambient_light")
        elif self.light_type == "directional":
            self.light = DirectionalLight("directional_light")
        self.light.setColor(self.color)

    def attach_to_render(self, render):
        """
        Attach the light to the render and set its position or orientation if applicable.
        """
        self.node_path = render.attachNewNode(self.light)
        if self.position:
            self.node_path.setPos(self.position)
        if self.orientation:
            self.node_path.setHpr(self.orientation)
        render.setLight(self.node_path)


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
        logging.debug("CameraController initialized with camera: {}".format(camera))

    def spin_camera(self, task):
        """
        Methodically rotate the camera around the scene based on the elapsed time, ensuring a continuous and smooth motion.

        Parameters:
        - task (Task): A task object that provides context, particularly the elapsed time since the task began.

        Returns:
        - Task.cont: A constant indicating that the task should continue running.
        """
        # Calculate the angle in degrees, adhering to a rotation rate of 6 degrees per second.
        angle_degrees = task.time * 6.0
        logging.debug("Calculated angle in degrees: {}".format(angle_degrees))

        # Convert the angle from degrees to radians for trigonometric calculations.
        angle_radians = angle_degrees * (np.pi / 180.0)
        logging.debug("Converted angle in radians: {}".format(angle_radians))

        # Calculate the camera's position using trigonometric functions to ensure a circular path.
        position_vector = np.array(
            [20 * np.sin(angle_radians), -20 * np.cos(angle_radians), 3]
        )
        logging.debug("Calculated position vector: {}".format(position_vector))

        # Set the camera's position using the calculated position vector.
        self.camera.setPos(tuple(position_vector))
        logging.info("Camera position set to: {}".format(tuple(position_vector)))

        # Orient the camera to look at the origin of the scene, ensuring it remains focused on the central point.
        self.camera.lookAt((0, 0, 0))
        logging.info("Camera oriented to look at the origin.")

        # Return the continuation constant to keep the task alive.
        return Task.cont


class ModelColorManager:
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
        # Increment the hue angle by 0.5 degrees per update cycle, wrapping around at 360 degrees using modular arithmetic.
        self.hue_angle = (self.hue_angle + 0.5) % 360
        logging.debug("Updated hue angle for color cycling: {}".format(self.hue_angle))

        # Convert the hue angle to an RGBA color with full opacity using a detailed conversion function.
        rgba_color = convert_hpr_to_vbase4_with_full_opacity(self.hue_angle[0], 0, 0)
        logging.debug("Converted RGBA color: {}".format(rgba_color))

        # Apply the new color to all geometry nodes within the render node.
        geom_nodes = self.render_node.findAllMatches("**/+GeomNode")
        for index, node in enumerate(geom_nodes):
            node.node().setAttrib(ColorAttrib.makeFlat(rgba_color))
            logging.info("Applied new color to node {}: {}".format(index, node))

        # Return the continuation constant to keep the task alive.
        return Task.cont


class ModelShowcase(ShowBase):
    """
    A class meticulously designed to render 3D models with dynamic lighting, color cycling, and camera manipulation,
    ensuring a high-quality visual presentation and interactive experience.
    """

    def __init__(self):
        """
        Initialize the ModelShowcase with enhanced rendering settings and tasks, setting up the environment for
        showcasing various 3D models with dynamic attributes.
        """
        super().__init__()
        self.model_index = 0
        self.model_names = self.retrieve_model_names()
        self.setup_lights()
        self.camera_controller = CameraController(self.camera)
        self.color_cycler = ColorCycler(self.render)

    def setup_lights(self):
        """
        Methodically setup all lights by creating and attaching them to the render, ensuring each light is configured
        with precise characteristics for optimal illumination.
        """
        light_attributes = np.array(
            [
                (1.0, 1.0, 1.0, 1.0, 10, 20, 0, None, "point"),
                (0.5, 0.5, 0.5, 1.0, None, None, None, None, "ambient"),
                (0.8, 0.8, 0.8, 1.0, None, None, None, (0, -60, 0), "directional"),
            ],
            dtype=[
                ("r", "f4"),
                ("g", "f4"),
                ("b", "f4"),
                ("a", "f4"),
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("orientation", "O"),
                ("type", "U10"),
            ],
        )

        for light_data in light_attributes:
            light = LightingComponent(
                light_type=light_data["type"],
                color=(
                    light_data["r"],
                    light_data["g"],
                    light_data["b"],
                    light_data["a"],
                ),
                position=(
                    (light_data["x"], light_data["y"], light_data["z"])
                    if not np.isnan(light_data["x"])
                    else None
                ),
                orientation=light_data["orientation"],
            )
            light.create_light()
            light.attach_to_render(self.render)

    def play(self):
        """
        Play the 3D model showcase with dynamic lighting and color cycling, initiating the main application loop.
        """
        self.taskMgr.add(self.camera_controller.spin_camera, "SpinCameraTask")
        self.taskMgr.add(self.color_cycler.update_color, "UpdateColorTask")
        self.run()

    def pause(self):
        """
        Pause the 3D model showcase, effectively stopping the dynamic tasks.
        """
        self.taskMgr.remove("SpinCameraTask")
        self.taskMgr.remove("UpdateColorTask")

    def restart(self):
        """
        Restart the 3D model showcase, resetting the dynamic attributes and resuming the tasks.
        """
        self.pause()
        self.play()

    def next_model(self):
        """
        Load the next 3D model in the showcase, cycling forward through the available models.
        """
        self.model_index = (self.model_index + 1) % len(self.model_names)
        self.load_model(self.model_names[self.model_index])

    def previous_model(self):
        """
        Load the previous 3D model in the showcase, cycling backward through the available models.
        """
        self.model_index = (self.model_index - 1) % len(self.model_names)
        self.load_model(self.model_names[self.model_index])

    def load_model(self, model_name):
        """
        Load a specific 3D model into the showcase, ensuring it is properly positioned and displayed within the render.
        """
        model = getattr(self, f"construct_{model_name}")()
        model.reparentTo(self.render)

    def clear_scene(self):
        """
        Clear the scene of all 3D models, ensuring a clean slate for loading new models.
        """
        self.render.removeNode()

    def retrieve_model_names(self):
        """
        Retrieve the available 3D models in the showcase, extracting names from the construction methods.
        """
        model_names = [name[10:] for name in dir(self) if name.startswith("construct_")]
        return model_names

    def run(self):
        """
        Run the Panda3D application, initiating the main event loop for the application.
        """
        self.taskMgr.run()
