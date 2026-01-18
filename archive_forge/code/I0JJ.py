import pygame
import random
import heapq
import logging
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Set

# Initialize logging with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_grid(size: int) -> np.ndarray:
    """
    Constructs a two-dimensional square grid of specified size where each cell within the grid is initialized to zero. This grid is represented as a NumPy array, which provides optimized operations and better performance for large grids.

    Args:
        size (int): The dimension of the grid, specifically indicating both the number of rows and the number of columns, as the grid is square.

    Returns:
        np.ndarray: A 2D NumPy array where each element is initialized to zero. The array encapsulates the complete grid structure.
    """
    grid = np.zeros((size, size), dtype=int)
    logging.debug(
        f"Grid of size {size}x{size} created with all elements initialized to zero using NumPy."
    )
    return grid


def convert_to_graph(grid: np.ndarray) -> nx.Graph:
    """
    Converts a two-dimensional grid into a graph representation using the NetworkX library, which provides extensive capabilities and optimized performance for graph operations. Each cell in the grid is treated as a node in the graph. The nodes are connected to their adjacent nodes (up, down, left, right) with edges that have randomly assigned weights.

    Args:
        grid (np.ndarray): A two-dimensional NumPy array representing the grid.

    Returns:
        nx.Graph: A NetworkX graph where each node represents a cell in the grid and edges connect adjacent nodes with random weights.
    """
    size = grid.shape[0]
    graph = nx.Graph()
    for i in range(size):
        for j in range(size):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    weight = random.random()
                    graph.add_edge((i, j), (ni, nj), weight=weight)
                    logging.debug(
                        f"Edge added from {(i, j)} to {(ni, nj)} with weight {weight} using NetworkX."
                    )
    return graph


def prim_mst(graph: nx.Graph, start_vertex: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Implements Prim's algorithm to compute the Minimum Spanning Tree (MST) from a given graph starting from a specified vertex using NetworkX's optimized algorithm. This method is a greedy algorithm that finds a minimum spanning tree for a weighted undirected graph.

    Args:
        graph (nx.Graph): A NetworkX graph representing the grid as a graph.
        start_vertex (Tuple[int, int]): The starting vertex from which the MST will begin to be computed.

    Returns:
        List[Tuple[int, int]]: A list of tuples representing the nodes that form the Minimum Spanning Tree, starting from the start_vertex.
    """
    mst = list(
        nx.algorithms.tree.mst.minimum_spanning_tree(
            graph, algorithm="prim", weight="weight", start=start_vertex
        ).nodes()
    )
    logging.info(
        f"MST completed with vertices: {mst} using NetworkX's Prim's algorithm."
    )
    return mst


def draw_path(
    screen,
    path: List[Tuple[int, int]],
    cell_size: int,
    current_index: int,
    max_length: int,
    frame_count: int,
):
    """
    Draw the path on the screen using Pygame with a gradient neon glow effect that smoothly transitions through a spectrum of colors. Each segment of the path also changes its color dynamically, creating a gradient of gradients effect. Additionally, implement a fading glow effect for the segments.

    Args:
        screen: Pygame screen object.
        path (List[Tuple[int, int]]): The path to draw.
        cell_size (int): Size of each cell in the grid.
        current_index (int): Current index in the path for the animation.
        max_length (int): Maximum number of segments to display at once.
        frame_count (int): Current frame count to adjust the color dynamically.
    """
    start = max(0, current_index - max_length)
    end = current_index + 1
    segments = path[start:end]

    def compute_color(index, frame_count):
        base_colors = [
            (0, 0, 0),
            (255, 0, 0),
            (255, 165, 0),
            (255, 255, 0),
            (0, 128, 0),
            (0, 0, 255),
            (75, 0, 130),
            (238, 130, 238),
            (255, 255, 255),
            (128, 128, 128),
            (0, 0, 0),
        ]
        num_colors = len(base_colors)
        base_index = index % num_colors
        next_index = (base_index + 1) % num_colors
        ratio = (index % num_colors) / num_colors
        r = int(
            base_colors[base_index][0] * (1 - ratio)
            + base_colors[next_index][0] * ratio
        )
        g = int(
            base_colors[base_index][1] * (1 - ratio)
            + base_colors[next_index][1] * ratio
        )
        b = int(
            base_colors[base_index][2] * (1 - ratio)
            + base_colors[next_index][2] * ratio
        )
        r = (r + 2 * frame_count) % 256
        g = (g + 2 * frame_count) % 256
        b = (b + 2 * frame_count) % 256
        return (r, g, b)

    for i, (x, y) in enumerate(segments):
        color = compute_color(i + frame_count, frame_count)
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, [min(c + 50, 255) for c in color], rect, 1)
        if i < len(segments) - 1:
            fade_color = [max(c - 10 * (len(segments) - i), 0) for c in color]
            pygame.draw.rect(screen, fade_color, rect, 1)
        logging.debug(f"Segment at {(x, y)} drawn with color {color}.")


def main():
    """
    The main function initializes the pygame window, sets up the game grid and graph, and handles the main game loop.
    """
    pygame.init()
    grid_size = 100
    cell_size = 5
    screen = pygame.display.set_mode(
        (grid_size * cell_size, grid_size * cell_size), pygame.RESIZABLE
    )
    pygame.display.set_caption("Dynamic Hamiltonian Cycle Visualization")

    logging.info("Creating grid and converting to graph using NumPy and NetworkX.")
    grid = create_grid(grid_size)
    graph = convert_to_graph(grid)
    current_path = prim_mst(graph, (0, 0))
    path_index = 0
    path_length = 1000

    running = True
    clock = pygame.time.Clock()
    while running:
        frame_count = pygame.time.get_ticks() // 10
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                logging.info("Quitting the game.")

        screen.fill((0, 0, 0))
        draw_path(screen, current_path, cell_size, path_index, path_length, frame_count)
        pygame.display.flip()
        path_index += 1

        if path_index >= len(current_path) - path_length // 2:
            new_start_vertex = current_path[-1]
            logging.debug(f"Recalculating path from vertex {new_start_vertex}.")
            current_path = current_path[:path_index] + prim_mst(graph, new_start_vertex)

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
