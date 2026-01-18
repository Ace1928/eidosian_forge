from typing import List, Tuple, Dict, Callable, Any, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import sys
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm  # Corrected import for colormap access
import sys
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Define type aliases for clarity
Point3D = Tuple[float, float, float]
Hexagon3D = List[Point3D]  # Includes center as the last point
StructureInfo = List[Tuple[Hexagon3D, List[int]]]  # Hexagon with its label
Label = List[int]

class Arrow3D(FancyArrowPatch):
    """
    A class for drawing 3D arrows in a matplotlib figure, extending FancyArrowPatch.
    """
    def __init__(self, xs: List[float], ys: List[float], zs: List[float], *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer: Optional[Any] = None) -> float:
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, renderer.M if renderer else plt.gca().get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def generate_hexagon(center: Point3D, side_length: float, elevation: float) -> Hexagon3D:
    """
    Generates the vertices of a 3D hexagon centered at `center`, including the center.
    """
    vertices = [center]  # Center for easier handling
    for i in range(6):
        angle_rad = 2 * math.pi / 6 * i
        x = center[0] + side_length * math.cos(angle_rad)
        y = center[1] + side_length * math.sin(angle_rad)
        vertices.append((x, y, elevation))
    return vertices

def hexagon_connections(hexagon: Hexagon3D, ax: plt.Axes, color: str):
    """
    Draws arrows between hexagon vertices to indicate directional connections.
    """
    # Outer vertices connections
    for i in range(1, 7):
        start = hexagon[i]
        for j in [1, 2, 3]:  # Direct connection, skip one, skip two
            end = hexagon[(i + j) % 6 or 6]  # Ensures wrapping around the hexagon
            arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                            mutation_scale=10, lw=1, arrowstyle="-|>", color=color)
            ax.add_artist(arrow)
        # Connection to the center
        center = hexagon[0]
        arrow = Arrow3D([start[0], center[0]], [start[1], center[1]], [start[2], center[2]],
                        mutation_scale=10, lw=1, arrowstyle="-|>", color=color)
        ax.add_artist(arrow)

def plot_3d_structure(structure: StructureInfo):
    """
    Plots the 3D structure with arrows to represent connections.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_map = plt.get_cmap('viridis')  # Corrected colormap access
    max_layers = max(hexagon[1][0] for hexagon in structure) + 1
    for hexagon, label in structure:
        layer = label[0]
        color = color_map(layer / float(max_layers))
        hexagon_connections(hexagon, ax, color=color)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def interactive_cli():
    """
    Interactive CLI for generating and visualizing the hexagonal structure.
    """
    try:
        neurons = int(input("Enter the number of neurons for the first layer: "))
        # This part of the code needs expansion to dynamically generate the structure based on `neurons`
        # For demonstration, this is a simplified version
        side_length = 1.0  # Assuming a constant side length for simplicity
        elevation = 0.0  # Starting elevation
        center = (0.0, 0.0, elevation)
        hexagon = generate_hexagon(center, side_length, elevation)
        structure = [(hexagon, [0, 0, 0, 0, 0, 0])]  # Simplified structure info

        # Plotting the structure
        plot_3d_structure(structure)
    except ValueError:
        print("Please enter a valid integer for the number of neurons.")
        sys.exit(1)

if __name__ == "__main__":
    interactive_cli()

