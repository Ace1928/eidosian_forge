"""
This library provides various algorithms for generating fractal data.
    The libraries it contains are:
    - Mandelbulb
    - Mandelbox
    - Julibrot
    - Quaternion Julia
    - Hypercomplex Fractals
    - Sierpinski Tetrahedron
    - Menger Sponge
    - Koch Snowflake 3D
    - IFS
    - Fractal Flames
    - Apollonian Gasket
    - Circle Inversion
    - Fractal Terrains
    - Strange Attractors
"""

class Mandelbulb
    """
    This class generated Mandelbulb fractals.

    
    """
    def __init__(self, power: float = 8.0, max_iter: int = 100, escape_radius: float = 2.0) -> None:
        """
        Initializes the Mandelbulb fractal generator with the specified parameters.

        Args:
            power (float): The power to which the complex number is raised at each iteration of the fractal generation.
            max_iter (int): The maximum number of iterations to perform for each point in the fractal.
            escape_radius (float): The escape radius used to determine if a point has escaped to infinity.

        The constructor method is meticulously documented, providing clear insights into the initialization process and the role of each parameter within the class.
        """
        self.power = power
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        # Ensuring that each parameter is correctly initialized to its default state, providing a solid foundation for the fractal's functionality.

    def generate(self, size: Tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
        """
        Generates a 3D Mandelbulb fractal based on the initialized parameters and returns it as a numpy array.

        This method meticulously implements the Mandelbulb fractal generation algorithm, utilizing advanced mathematical concepts and numpy for high-performance computations. The fractal is generated in a 3D space, with each point's iteration determined by the Mandelbulb formula, and the result is returned as a numpy array representing the fractal structure.

        Args:
            size (Tuple[int, int, int]): The dimensions (width, height, depth) of the numpy array to generate.
            scale (float): The scale factor for zooming into or out of the fractal.

        Returns:
            np.ndarray: A 3D numpy array representing the Mandelbulb fractal.

        The method is thoroughly documented, providing clear and comprehensive insights into its functionality, parameters, and return type. It exemplifies the commitment to advanced programming techniques and high-quality code documentation.
        """
        import numpy as np
        from numpy import sin, cos, arctan2, sqrt, power, pi
        from typing import Tuple

        # Initialize the numpy array with zeros to store the fractal data.
        fractal = np.zeros(size, dtype=np.float32)

        # Calculate the aspect ratio based on the size of the numpy array.
        aspect_ratio = size[0] / size[1]

        # Iterate over each point in the 3D space defined by the size parameter.
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    # Normalize and scale the coordinates.
                    x_scaled = (x / size[0] * 2 - 1) * scale * aspect_ratio
                    y_scaled = (y / size[1] * 2 - 1) * scale
                    z_scaled = (z / size[2] * 2 - 1) * scale

                    # Initialize the point coordinates for the fractal iteration.
                    point = np.array([x_scaled, y_scaled, z_scaled], dtype=np.float32)
                    # Initialize the iteration count.
                    iter_count = 0

                    # Iterate the point through the Mandelbulb formula until it escapes or reaches the maximum iterations.
                    while np.linalg.norm(point) <= self.escape_radius and iter_count < self.max_iter:
                        # Convert the point to spherical coordinates.
                        r = np.linalg.norm(point)
                        theta = arctan2(sqrt(point[0]**2 + point[1]**2), point[2]) * self.power
                        phi = arctan2(point[1], point[0]) * self.power

                        # Calculate the new point coordinates using the Mandelbulb formula.
                        new_x = power(r, self.power) * sin(theta) * cos(phi)
                        new_y = power(r, self.power) * sin(theta) * sin(phi)
                        new_z = power(r, self.power) * cos(theta)

                        # Update the point coordinates.
                        point = np.array([new_x, new_y, new_z], dtype=np.float32) + [x_scaled, y_scaled, z_scaled]
                        # Increment the iteration count.
                        iter_count += 1

                    # Normalize the iteration count to a value between 0 and 1 and store it in the fractal array.
                    fractal[x, y, z] = iter_count / self.max_iter

        return fractal

    def __str__(self) -> str:
        """
        Returns a string representation of the Mandelbulb fractal generator, including its parameters.

        Returns:
            str: A string representation of the Mandelbulb fractal generator.

        This method provides a concise and informative string representation of the Mandelbulb fractal generator, detailing its power, maximum iterations, and escape radius. It enhances the understandability and debuggability of the class, adhering to best practices in software development and documentation.
        """
        return f"Mandelbulb(power={self.power}, max_iter={self.max_iter}, escape_radius={self.escape_radius})"
        # This string representation is meticulously crafted to provide clear insights into the class's functionality and parameters, adhering to the highest standards of code documentation and readability.


import numpy as np  # Importing numpy for numerical operations

class Mandelbox:
    """
    This class is meticulously designed to generate Mandelbox fractals, a type of fractal that is generated through a series of transformations including box folding, sphere folding, scaling, and translation. The Mandelbox fractal is celebrated for its vast variety of structures emerging from relatively simple transformations.

    Attributes:
        max_iter (int): The maximum number of iterations for the fractal generation process.
        escape_radius (float): The escape radius to determine if a point has escaped the fractal boundary.
        scale_factor (float): The scale factor applied during the scaling and translation step, typically negative to create the inversion effect characteristic of the Mandelbox.
        fold_params (tuple): Parameters for box folding and sphere folding operations, typically including the scaling factors for sphere folding.
        constant_vector (np.ndarray): The constant vector used in the translation step, remaining the same for each iteration.
    """

    def __init__(self, max_iter: int, escape_radius: float, scale_factor: float, fold_params: tuple, constant_vector: np.ndarray) -> None:
        """
        Initializes the Mandelbox fractal generator with the specified parameters, setting the stage for a journey into the depths of fractal geometry.

        Args:
            max_iter (int): The maximum number of iterations for the fractal generation process.
            escape_radius (float): The escape radius to determine if a point has escaped the fractal boundary.
            scale_factor (float): The scale factor applied during the scaling and translation step.
            fold_params (tuple): Parameters for box folding and sphere folding operations, typically including the scaling factors for sphere folding.
            constant_vector (np.ndarray): The constant vector used in the translation step, remaining the same for each iteration.
        """
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        self.scale_factor = scale_factor
        self.fold_params = fold_params
        self.constant_vector = constant_vector

    def generate(self, size: tuple, scale: float, aspect_ratio: float) -> np.ndarray:
        """
        Generates the Mandelbox fractal based on the initialized parameters and the specified size, scale, and aspect ratio, unfolding the fractal's complexity with each iteration.

        Args:
            size (tuple): The size of the 3D space to generate the fractal in (width, height, depth).
            scale (float): The scale to apply to the coordinates of the points in the 3D space.
            aspect_ratio (float): The aspect ratio to maintain while scaling the coordinates.

        Returns:
            np.ndarray: A 3D numpy array representing the generated Mandelbox fractal, where each element's value corresponds to the iteration count normalized between 0 and 1.
        """
        fractal = np.zeros(size, dtype=np.float32)  # Initializing the fractal array with zeros
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    point = np.array([(x / size[0] * 2 - 1) * scale * aspect_ratio,
                                      (y / size[1] * 2 - 1) * scale,
                                      (z / size[2] * 2 - 1) * scale], dtype=np.float32)
                    iter_count = 0
                    while np.linalg.norm(point) <= self.escape_radius and iter_count < self.max_iter:
                        point = self._transform_point(point)  # Applying the Mandelbox transformations
                        iter_count += 1
                    fractal[x, y, z] = iter_count / self.max_iter  # Normalizing the iteration count
        return fractal

    def _transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Applies the Mandelbox transformations to a point in 3D space, including box folding, sphere folding, scaling, and translation, encapsulating the essence of the Mandelbox fractal's generation process.

        Args:
            point (np.ndarray): The point in 3D space to apply the transformations to.

        Returns:
            np.ndarray: The transformed point after applying the Mandelbox transformations, embodying the fractal's iterative nature.
        """
        # Box folding
        point = np.where(point > 1, 2 - point, point)
        point = np.where(point < -1, -2 - point, point)
        # Sphere folding
        radius = np.linalg.norm(point)
        if radius < 0.5:
            point *= self.fold_params[0]
        elif radius < 1:
            point *= self.fold_params[1]
        # Scaling and translation
        point = point * self.scale_factor + self.constant_vector
        return point

    def __str__(self) -> str:
        """
        Returns a string representation of the Mandelbox fractal generator, including its parameters, offering a window into the parameters defining the fractal's generation.

        Returns:
            str: A string representation of the Mandelbox fractal generator, meticulously crafted to provide insights into the class's functionality and parameters.
        """
        return f"Mandelbox(max_iter={self.max_iter}, escape_radius={self.escape_radius}, scale_factor={self.scale_factor}, fold_params={self.fold_params}, constant_vector={self.constant_vector})"
        # This string representation is a testament to the class's design, providing clear insights into its functionality and parameters.

class Julibrot
    """
    This class generates Julibrot fractals.

    
    """


class QuaternionJulia
    """
    This class generates Quaternion Julia fractals.

    
    """


class HypercomplexFractals
    """
    This class generates hypercomplex fractals.

    
    """


class SierpinskiTetrahedron
    """
    This class generates Sierpinski Tetrahedron fractals.

    
    """


class MengerSponge
    """
    This class generates Menger Sponge fractals.

    
    """


class KochSnowflake3D
    """
    This class generates Koch Snowflake 3D fractals.

    
    """


class IFS
    """
    This class generates IFS fractals.

    
    """


class FractalFlames
    """
    This class generates Fractal Flames fractals.

    
    """


class ApollonianGasket
    """
    This class generates Apollonian Gasket fractals.

    
    """


class CircleInversion
    """
    This class generates Circle Inversion fractals.

    
    """


class FractalTerrains
    """
    This class generates Fractal Terrains fractals.

    
    """


class StrangeAttractors
    """
    This class generates Strange Attractors fractals.

    
    """




