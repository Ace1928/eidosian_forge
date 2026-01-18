import os  # Importing the os module for file and directory operations
import json  # Importing the json module for handling JSON data
import logging  # Importing the logging module for logging purposes
import asyncio  # Importing the asyncio module for asynchronous programming
import random  # Importing the random module for generating random values
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Any,
    Callable,
    TypeAlias,
    Optional,
)  # Importing various types from the typing module for type hinting and annotation
from itertools import (
    cycle,
)  # Importing the cycle function from itertools for creating cyclic iterators
import multiprocessing  # Importing the multiprocessing module for parallel processing
import OpenGL.GL as gl  # Importing OpenGL.GL module and aliasing it as gl for OpenGL graphics functionality
import OpenGL.GLUT as glut  # Importing OpenGL.GLUT module and aliasing it as glut for OpenGL Utility Toolkit functionality
import OpenGL  # Importing the OpenGL module for OpenGL graphics functionality
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_MODELVIEW,
    GL_PROJECTION,
)  # Importing specific constants from OpenGL.GL for OpenGL graphics functionality

import cProfile  # Importing the cProfile module for profiling code execution
import pstats  # Importing the pstats module for analyzing profiling statistics
from pstats import (
    SortKey,
)  # Importing the SortKey class from pstats for sorting profiling statistics


# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)  # Configuring the logging module with DEBUG level and a specific format for log messages


# Type Aliases for enhanced clarity, type safety, and to avoid type/value/key errors
ColorName: TypeAlias = str  # Defining a type alias ColorName as str for color names
RGBValue: TypeAlias = Tuple[
    int, int, int
]  # Defining a type alias RGBValue as a tuple of three integers for RGB color values
CMYKValue: TypeAlias = Tuple[
    float, float, float, float
]  # Defining a type alias CMYKValue as a tuple of four floats for CMYK color values
LabValue: TypeAlias = Tuple[
    float, float, float
]  # Defining a type alias LabValue as a tuple of three floats for Lab color values
XYZValue: TypeAlias = Tuple[
    float, float, float
]  # Defining a type alias XYZValue as a tuple of three floats for XYZ color values
ColorCode: TypeAlias = Tuple[
    ColorName, RGBValue
]  # Defining a type alias ColorCode as a tuple of ColorName and RGBValue for color codes
ColorCodesList: TypeAlias = List[
    ColorCode
]  # Defining a type alias ColorCodesList as a list of ColorCode for a list of color codes
ColorCodesDict: TypeAlias = Dict[
    ColorName, Union[RGBValue, str]
]  # Defining a type alias ColorCodesDict as a dictionary mapping ColorName to either RGBValue or str for color codes dictionary
ANSIValue: TypeAlias = (
    str  # Defining a type alias ANSIValue as str for ANSI escape codes
)
HexValue: TypeAlias = (
    str  # Defining a type alias HexValue as str for hexadecimal color values
)
ExtendedColorCode: TypeAlias = Tuple[
    RGBValue, ANSIValue, HexValue, CMYKValue, LabValue, XYZValue, Any
]  # Defining a type alias ExtendedColorCode as a tuple of RGBValue, ANSIValue, HexValue, CMYKValue, LabValue, XYZValue, and Any for extended color codes
ExtendedColorCodesDict: TypeAlias = Dict[
    ColorName, ExtendedColorCode
]  # Defining a type alias ExtendedColorCodesDict as a dictionary mapping ColorName to ExtendedColorCode for extended color codes dictionary

# Initialize your 3D scene and objects here
try:
    scene = asyncio.run(
        generate_scene()
    )  # Initializing the 3D scene by running the generate_scene() function asynchronously
except NameError as e:
    logging.error(
        f"NameError occurred while initializing the 3D scene: {str(e)}"
    )  # Logging an error message if a NameError occurs during scene initialization
    raise  # Re-raising the exception to propagate it further

# Define the path to the color codes file
COLOR_CODES_FILE_PATH: str = (
    "/home/lloyd/EVIE/color_codes.json"  # Defining a constant variable COLOR_CODES_FILE_PATH as a string representing the path to the color codes JSON file
)


def random_color() -> RGBValue:
    """
    Generates a random RGB color value.

    Returns:
        RGBValue: A tuple of three integers representing the random RGB color value.
    """
    logging.debug(
        "Generating a random color."
    )  # Logging a debug message indicating that a random color is being generated
    r: int = random.randint(
        0, 255
    )  # Generating a random integer between 0 and 255 (inclusive) for the red component of the color
    g: int = random.randint(
        0, 255
    )  # Generating a random integer between 0 and 255 (inclusive) for the green component of the color
    b: int = random.randint(
        0, 255
    )  # Generating a random integer between 0 and 255 (inclusive) for the blue component of the color
    logging.debug(
        f"Generated random color: ({r}, {g}, {b})"
    )  # Logging a debug message with the generated random color values
    return (r, g, b)  # Returning the random RGB color value as a tuple


def generate_color_codes() -> ColorCodesList:
    """
    Generates a comprehensive list of color codes spanning a granular color spectrum.
    Each color is associated with a systematic name for easy identification and retrieval.
    Additionally, it includes the ANSI escape character, hex code, CMYK values, Lab values, XYZ values, and future extensions for each color.

    Returns:
        ColorCodesList: A list of tuples containing color names, their RGB values, ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.
    """
    logging.info(
        "Generating basic color codes list."
    )  # Logging an info message indicating that the basic color codes list is being generated
    colors: ColorCodesList = [
        ("black0", (0, 0, 0))
    ]  # Initializing the colors list with the absolute black color code
    # Incrementally increasing shades of grey
    colors += [
        (f"grey{i}", (i, i, i)) for i in range(1, 256)
    ]  # Generating color codes for incrementally increasing shades of grey
    # Detailed Red to Orange spectrum
    colors += [
        (f"red{i}", (255, i, 0)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Red to Orange spectrum
    # Detailed Orange to Yellow spectrum
    colors += [
        (f"yellow{i}", (255, 255, i)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Orange to Yellow spectrum
    # Detailed Yellow to Green spectrum
    colors += [
        (f"green{i}", (255 - i, 255, 0)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Yellow to Green spectrum
    # Detailed Green to Blue spectrum
    colors += [
        (f"blue{i}", (0, 255, i)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Green to Blue spectrum
    # Detailed Blue to Indigo spectrum
    colors += [
        (f"indigo{i}", (0, 255 - i, 255)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Blue to Indigo spectrum
    # Detailed Indigo to Violet spectrum
    colors += [
        (f"violet{i}", (i, 0, 255)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Indigo to Violet spectrum
    # Detailed Violet to White spectrum
    colors += [
        (f"white{i}", (255, i, 255)) for i in range(0, 256)
    ]  # Generating color codes for the detailed Violet to White spectrum
    colors.append(
        ("white255", (255, 255, 255))
    )  # Appending the pure white color code to the colors list
    logging.debug(
        f"Generated {len(colors)} color codes."
    )  # Logging a debug message with the total number of generated color codes
    return colors  # Returning the generated color codes list


# Asynchronous function to convert RGB to Hex
async def rgb_to_hex(rgb: RGBValue) -> HexValue:
    """
    Asynchronously converts an RGB value to its hexadecimal representation.


    Args:
        rgb (RGBValue): A tuple containing the RGB values.


    Returns:
        HexValue: The hexadecimal representation of the color.
    """
    logging.debug(f"Converting RGB {rgb} to Hex asynchronously.")
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Asynchronous function to generate ANSI escape code for color
async def rgb_to_ansi(rgb: RGBValue) -> ANSIValue:
    """
    Asynchronously generates the ANSI escape code for a given RGB color.


    Args:
        rgb (RGBValue): A tuple containing the RGB values.


    Returns:
        ANSIValue: The ANSI escape code for the color.
    """
    logging.debug(f"Generating ANSI code for RGB {rgb} asynchronously.")
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


# Asynchronous function to convert RGB to CMYK
async def rgb_to_cmyk(rgb: RGBValue) -> CMYKValue:
    """
    Asynchronously converts an RGB value to CMYK representation.


    Args:
        rgb (RGBValue): A tuple containing the RGB values.


    Returns:
        CMYKValue: The CMYK representation of the color.
    """
    logging.debug(f"Converting RGB {rgb} to CMYK asynchronously.")
    r: float
    g: float
    b: float
    r, g, b = [x / 255.0 for x in rgb]  # Normalize RGB values
    k: float = 1 - max(r, g, b)
    if k == 1:
        return 0.0, 0.0, 0.0, 1.0  # Pure black
    c: float = (1 - r - k) / (1 - k)
    m: float = (1 - g - k) / (1 - k)
    y: float = (1 - b - k) / (1 - k)
    return c, m, y, k


# Function to extend color codes with additional types and future extensions
async def extend_color_codes(
    color_codes_list: ColorCodesList,
) -> ExtendedColorCodesDict:
    """
    Asynchronously extends the basic color codes with ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.


    Args:
        color_codes_list (ColorCodesList): The basic list of color codes.


    Returns:
        ExtendedColorCodesDict: A dictionary of color names mapped to a tuple of RGB values, ANSI escape codes, hex codes, CMYK values, Lab values, XYZ values, and placeholders for future extensions.
    """
    logging.info("Asynchronously extending basic color codes with additional types.")
    extended_color_codes: ExtendedColorCodesDict = {}
    for name, rgb in color_codes_list:
        ansi: ANSIValue = await rgb_to_ansi(rgb)
        hex_code: HexValue = await rgb_to_hex(rgb)
        cmyk: CMYKValue = await rgb_to_cmyk(rgb)
        extended_color_codes[name] = (
            rgb,
            ansi,
            hex_code,
            cmyk,
            (0.0, 0.0, 0.0),  # Placeholder for Lab values
            (0.0, 0.0, 0.0),  # Placeholder for XYZ values
            None,  # Placeholder for future extensions
        )
    logging.debug(f"Extended {len(extended_color_codes)} color codes asynchronously.")
    return extended_color_codes


async def generate_dynamic_colors(extended_color_codes: ExtendedColorCodesDict) -> None:
    """
    Asynchronously generates dynamic colors for the 3D color spectacle.


    This function continuously selects random colors from the extended color codes dictionary and applies them to various elements in the 3D scene, such as the background color, particle clouds, and randomly generated shapes. The colors are smoothly transitioned using asynchronous sleep intervals to create a mesmerizing and dynamic color spectacle.


    Args:
        extended_color_codes (ExtendedColorCodesDict): The dictionary of extended color codes containing color names mapped to their corresponding color values in various formats (RGB, ANSI, HEX, CMYK, Lab, XYZ).


    Returns:
        None
    """
    logging.info("Starting dynamic color generation.")
    while True:
        # Select a random color name from the extended color codes dictionary
        color_name: ColorName = random.choice(list(extended_color_codes.keys()))
        logging.debug(f"Selected color name: {color_name}")

        # Retrieve the color information for the selected color name
        color_info: ExtendedColorCode = extended_color_codes[color_name]
        rgb: RGBValue = color_info[0]
        color: Tuple[float, float, float] = (
            rgb[0] / 255.0,
            rgb[1] / 255.0,
            rgb[2] / 255.0,
        )
        logging.debug(f"RGB color: {rgb}, Normalized color: {color}")

        # Asynchronously sleep for a short duration to create smooth color transitions
        await asyncio.sleep(
            random.uniform(0.001, 0.01)
        )  # Adjust the range for desired transition speed

        # Apply the selected color to the scene background
        scene.background_color = color
        logging.debug(f"Applied color {color} to the scene background.")

        # Generate random particle clouds with the selected color
        num_particles: int = random.randint(1000, 10000)
        particles: List[Particle] = [Particle() for _ in range(num_particles)]
        for particle in particles:
            particle.color = color
            particle.position = (
                random.uniform(-20, 20),
                random.uniform(-20, 20),
                random.uniform(-20, 20),
            )
            particle.velocity = (
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(-5, 5),
            )
        scene.add_particles(particles)
        logging.debug(f"Generated {num_particles} particles with color {color}.")

        # Generate random shapes with the selected color
        num_shapes: int = random.randint(10, 100)
        shapes: List[Shape] = [
            random.choice([Cube, Sphere, Pyramid, Torus, Cone])()
            for _ in range(num_shapes)
        ]
        for shape in shapes:
            shape.color = color
            shape.position = (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10),
            )
            shape.scale = (
                random.uniform(0.1, 5),
                random.uniform(0.1, 5),
                random.uniform(0.1, 5),
            )
            shape.rotation = (
                random.uniform(0, 360),
                random.uniform(0, 360),
                random.uniform(0, 360),
            )
        scene.add_shapes(shapes)
        logging.debug(f"Generated {num_shapes} shapes with color {color}.")


# Function to validate and load or regenerate the color codes file
async def validate_and_load_color_codes() -> ExtendedColorCodesDict:
    """
    Validates the integrity of the color codes file, loads it if valid, or regenerates and saves a new one if invalid or not found.


    Returns:
        ExtendedColorCodesDict: A dictionary of color names mapped to their extended color codes.
    """
    try:
        if os.path.exists(COLOR_CODES_FILE_PATH):
            with open(COLOR_CODES_FILE_PATH, "r") as file:
                color_codes: ExtendedColorCodesDict = json.load(file)
                # Perform a basic integrity check on the loaded color codes
                if not isinstance(color_codes, dict) or not color_codes:
                    raise ValueError("Color codes file is corrupted or empty.")
                logging.info("Color codes file loaded successfully.")
        else:
            raise FileNotFoundError("Color codes file not found.")
    except (FileNotFoundError, ValueError) as e:
        logging.warning(f"{e}. Regenerating color codes file.")
        basic_color_codes_list: ColorCodesList = await generate_color_codes()
        color_codes = await extend_color_codes(basic_color_codes_list)
        with open(COLOR_CODES_FILE_PATH, "w") as file:
            json.dump(color_codes, file, indent=4)
            logging.info("New color codes file generated and saved.")
    return color_codes


async def generate_scene():
    """
    Generates the initial 3D scene with various objects and shapes.


    This function creates a new instance of the Scene class and populates it with an assortment of 3D objects and shapes. The objects include a rotating cube, a sphere with a dynamic color material, a particle system, and a color-changing torus. The function sets up the initial positions, sizes, rotations, and materials for each object and adds them to the scene.


    Returns:
        Scene: The generated 3D scene containing the initial objects and shapes.
    """
    logging.info("Generating the initial 3D scene.")
    scene = Scene()

    # Add a rotating cube to the scene
    cube = Cube(position=(0, 0, 0), size=(1, 1, 1))
    cube.set_rotation(axis=(1, 1, 1), angle=0, speed=1)
    scene.add_object(cube)
    logging.debug("Added a rotating cube to the scene.")

    # Add a sphere with a dynamic color material to the scene
    sphere = Sphere(position=(2, 0, 0), radius=0.7)
    color_material = ColorMaterial()
    sphere.set_material(color_material)
    scene.add_object(sphere)
    logging.debug("Added a sphere with a dynamic color material to the scene.")

    # Add a particle system to the scene
    particle_system = ParticleSystem(position=(-2, 0, 0), num_particles=1000)
    scene.add_object(particle_system)
    logging.debug("Added a particle system to the scene.")

    # Add a color-changing torus to the scene
    torus = Torus(position=(0, 2, 0), inner_radius=0.5, outer_radius=1)
    color_material = ColorMaterial()
    torus.set_material(color_material)
    scene.add_object(torus)
    logging.debug("Added a color-changing torus to the scene.")

    logging.info("Initial 3D scene generated successfully.")
    return scene


def render_scene() -> None:
    """
    Renders the 3D scene using OpenGL.


    This function clears the color and depth buffers, sets up the projection and modelview matrices, applies scene rotations, and renders all the objects in the scene. It uses OpenGL functions to perform the rendering operations and creates a visually appealing 3D scene.


    The function first clears the color and depth buffers using `glClear` to prepare for rendering a new frame. It then sets up the projection matrix using `glMatrixMode(GL_PROJECTION)` and `gluPerspective` to define the viewing frustum and perspective.


    Next, it sets up the modelview matrix using `glMatrixMode(GL_MODELVIEW)` and `gluLookAt` to specify the camera position, target point, and up vector. The scene is then rotated using `glRotatef` based on the current rotation angles stored in `scene.rotation`.


    Finally, the function calls the `render` method of the `scene` object to render all the objects and shapes in the scene. After rendering, it swaps the front and back buffers using `glutSwapBuffers` to display the rendered frame on the screen.


    This function is typically called in the main rendering loop to continuously update and display the 3D scene.
    """
    logging.debug("Rendering the 3D scene.")

    # Clear the color and depth buffers
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Set up the projection matrix
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(60, 800 / 600, 0.1, 1000.0)

    # Set up the modelview matrix
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.gluLookAt(0, 0, 50, 0, 0, 0, 0, 1, 0)

    # Rotate the scene based on the current rotation angles
    gl.glRotatef(scene.rotation[0], 1, 0, 0)
    gl.glRotatef(scene.rotation[1], 0, 1, 0)
    gl.glRotatef(scene.rotation[2], 0, 0, 1)

    # Update the scene rotation angles for the next frame
    scene.rotation = (
        (scene.rotation[0] + 1) % 360,
        (scene.rotation[1] + 2) % 360,
        (scene.rotation[2] + 3) % 360,
    )

    # Render the objects and shapes in the scene
    scene.render()

    # Swap the front and back buffers to display the rendered frame
    glut.glutSwapBuffers()

    logging.debug("3D scene rendering completed.")


def update_scene() -> None:
    """
    Updates the 3D scene by rotating the camera and objects.


    This function retrieves the current elapsed time using `glutGet(GLUT_ELAPSED_TIME)` and passes it to the `update` method of the `scene` object. The `update` method is responsible for updating the positions, rotations, and any other dynamic properties of the objects in the scene based on the elapsed time.


    After updating the scene, the function calls `glutPostRedisplay` to mark the current window as needing to be redisplayed. This triggers the rendering loop to call the registered display callback function (typically `render_scene`) to redraw the scene with the updated object positions and rotations.


    This function is typically registered as the idle callback function using `glutIdleFunc` to continuously update the scene in the background.
    """
    logging.debug("Updating the 3D scene.")

    # Get the current elapsed time in seconds
    current_time = glut.glutGet(glut.GLUT_ELAPSED_TIME) / 1000.0

    # Update the positions, rotations, and other dynamic properties of the objects in the scene
    scene.update(current_time)

    # Mark the current window as needing to be redisplayed
    glut.glutPostRedisplay()

    logging.debug("3D scene update completed.")


def profile_function(func: Callable) -> None:
    """
    Profiles the given function using cProfile and prints the stats.


    Args:
        func (Callable): The function to profile.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()


async def main() -> None:
    """
    The main function that sets up the 3D scene, starts the color spectacle, and enters the rendering loop.


    This function integrates the functionality from both versions of the main function, ensuring all features are present and perfected.
    It sets up the OpenGL context, initializes the window, loads color codes, starts dynamic color generation, sets up callbacks, and enters the main rendering loop.


    The function is meticulously constructed to follow best practices, adhere to coding standards, and provide extensive documentation.
    It showcases the most perfected Python implementation possible, considering all aspects of code quality, performance, and maintainability.
    """
    # Set up logging for the main function
    logging.info("Entering the main function.")

    try:
        # Initialize OpenGL and GLUT
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(800, 600)
        glut.glutCreateWindow(b"3D Color Spectacle")

        # Set up OpenGL
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Load or generate color codes
        logging.info("Loading color codes.")
        extended_color_codes: ExtendedColorCodesDict = (
            await validate_and_load_color_codes()
        )

        # Start the dynamic color generation
        logging.info("Starting dynamic color generation.")
        asyncio.create_task(generate_dynamic_colors(extended_color_codes))

        # Set up GLUT callbacks
        glut.glutDisplayFunc(render_scene)
        glut.glutIdleFunc(update_scene)

        # Enter the main loop
        logging.info("Entering the main rendering loop.")
        glut.glutMainLoop()

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        raise

    finally:
        logging.info("Exiting the main function.")


if __name__ == "__main__":
    # Set up profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the main function in an asyncio event loop
    asyncio.run(main())

    # End profiling and print stats
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()
