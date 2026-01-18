import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np
import pandas as pd
class HexagonalStructure:
    """
    Manages the generation, manipulation, and visualization of a 3D hexagonal structure.

    Attributes:
        layers (int): The number of layers in the hexagonal structure.
        side_length (float): The side length of each hexagon in the structure.
        structure (Structure3D): The generated 3D structure stored as a dictionary.
    """

    def __init__(self, layers: int, side_length: float) -> None:
        logging.info(f'Creating HexagonalStructure with {layers} layers and side length {side_length}')
        self.layers: int = layers
        self.side_length: float = side_length
        self.structure: Structure3D = self._generate_3d_structure()

    def _generate_hexagon(self, center: Point3D, elevation: float) -> Hexagon3D:
        logging.debug(f'Starting hexagon generation with center={center} and elevation={elevation}')
        vertices: Hexagon3D = []
        for i in range(6):
            angle_rad = 2 * math.pi / 6 * i
            x = center[0] + self.side_length * math.cos(angle_rad)
            y = center[1] + self.side_length * math.sin(angle_rad)
            vertices.append((x, y, elevation))
        logging.debug(f'Hexagon generated with vertices={vertices}')
        return vertices

    def _generate_3d_structure(self) -> Structure3D:
        """
        Generates a sophisticated 3D structure consisting of meticulously stacked hexagons,
        each placed with precision to form a cohesive, extended hexagonal prism structure. 
        This method embodies the pinnacle of algorithmic design, pushing the boundaries 
        of computational geometry to create a visually stunning and mathematically robust 
        representation of a hexagonal structure in three dimensions.

        Returns:
            Structure3D: A meticulously curated dictionary. Each key represents a layer index,
            associated with a list of hexagons within that layer. Each hexagon is further
            represented as a list of 3D points, constituting a comprehensive model of the 
            entire 3D hexagonal architecture.
        """
        logging.debug('Starting 3D structure generation')
        structure: Structure3D = {}
        elevation: float = 0.0
        elevation_step: float = self.side_length * math.sqrt(3) / 2
        for layer in range(self.layers):
            logging.debug(f'Generating layer {layer}')
            hexagons = []
            center_offset_x: float = self.side_length * 1.5 * layer
            center_offset_y: float = self.side_length * math.sqrt(3) / 2 * layer
            if layer == 0:
                base_center: Point3D = (0.0, 0.0, elevation)
                hexagons.append(self._generate_hexagon(base_center, elevation))
            else:
                for layer in range(1, self.layers):
                    elevation += elevation_step
                    previous_layer_hexagons = structure[layer - 1]
                    current_layer_hexagons = []
                    for hexagon in structure[layer - 1]:
                        for i in range(6):
                            angle_rad = math.pi / 3 * i
                            x = hexagon[0][0] + self.side_length * math.cos(angle_rad)
                            y = hexagon[0][1] + self.side_length * math.sin(angle_rad)
                            new_center: Point3D = (x, y, elevation)
                            if not any((np.allclose(new_center, h[0], atol=1e-08) for h in hexagons)):
                                hexagons.append(self._generate_hexagon(new_center, elevation))
            hexagons_centered: Hexagon3D = []
            for hexagon in hexagons:
                hexagon_centered: Hexagon3D = [(x - center_offset_x, y - center_offset_y, z) for x, y, z in hexagon]
                hexagons_centered.append(hexagon_centered)
            structure[layer] = hexagons_centered
            elevation += elevation_step
            logging.info(f'Hexagonal layer {layer} generated with {len(hexagons_centered)} hexagons.')
            logging.debug(f'Layer {layer} generation completed with {len(hexagons_centered)} hexagons')
        logging.debug('The 3D hexagonal structure has been fully realized to the zenith of algorithmic artistry.')
        logging.debug('3D structure generation completed')
        return structure

    def plot_structure(self) -> None:
        logging.info('Plotting 3D hexagonal structure')
        logging.debug('Starting structure plotting')
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(111, projection='3d')
        color_map = plt.get_cmap('viridis')
        for layer, hexagons in self.structure.items():
            color = color_map(layer / self.layers)
            for hexagon in hexagons:
                self.hexagon_connections(hexagon, ax, color=color)
                xs, ys, zs = zip(*hexagon)
                ax.plot(xs, ys, zs, color=color)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('3D Hexagonal Structure')
        plt.tight_layout()
        plt.show()
        logging.debug('Structure plotting completed')

    def hexagon_connections(self, hexagon: Hexagon3D, ax: plt.Axes, color: str) -> None:
        logging.debug(f'Drawing connections for hexagon with center={np.mean(np.array(hexagon), axis=0)}')
        for i in range(6):
            start = hexagon[i]
            for j in [1, 2, 3]:
                end = hexagon[(i + j) % 6]
                ax.add_artist(Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], mutation_scale=10, lw=1, arrowstyle='-|>', color=color))
        center: Point3D = np.mean(np.array(hexagon), axis=0)
        for vertex in hexagon:
            ax.add_artist(Arrow3D([vertex[0], center[0]], [vertex[1], center[1]], [vertex[2], center[2]], mutation_scale=10, lw=1, arrowstyle='-|>', color=color))
        logging.debug('Connections drawn')

    def export_to_csv(self, filename: str='hexagonal_structure.csv') -> None:
        logging.info(f'Exporting structure data to {filename}')
        logging.debug(f'Starting export of structure data to {filename}')
        data: List[Dict[str, Union[int, float]]] = []
        for layer, hexagons in self.structure.items():
            for hexagon in hexagons:
                for vertex in hexagon:
                    data.append({'Layer': layer, 'X': vertex[0], 'Y': vertex[1], 'Z': vertex[2]})
        df: pd.DataFrame = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f'Data exported successfully to {filename}')
        logging.debug(f'Data export to {filename} completed')