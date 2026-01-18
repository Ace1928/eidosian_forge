import pygame
import random
from typing import List, Tuple, Dict, Set


def create_grid(size: int) -> List[List[int]]:
    """
    Create a square grid initialized to zero.

    Args:
        size (int): The dimension of the grid (size x size).

    Returns:
        List[List[int]]: A 2D list representing the grid.
    """
    return [[0 for _ in range(size)] for _ in range(size)]


def convert_to_graph(
    grid: List[List[int]],
) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
    """
    Convert a grid to a graph with weighted edges between adjacent nodes.

    Args:
        grid (List[List[int]]): The input grid.

    Returns:
        Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]: Graph representation.
    """
    size = len(grid)
    graph = {}
    for i in range(size):
        for j in range(size):
            graph[(i, j)] = []
            if i < size - 1:
                graph[(i, j)].append(((i + 1, j), random.random()))
            if j < size - 1:
                graph[(i, j)].append(((i, j + 1), random.random()))
            if i > 0:
                graph[(i, j)].append(((i - 1, j), random.random()))
            if j > 0:
                graph[(i, j)].append(((i, j - 1), random.random()))
    return graph


def prim_mst(
    graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start_vertex: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Implement Prim's algorithm to compute the Minimum Spanning Tree (MST).

    Args:
        graph (Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]): Graph of nodes.
        start_vertex (Tuple[int, int]): Starting vertex for MST computation.

    Returns:
        List[Tuple[int, int]]: Sequence of nodes forming the MST.
    """
    visited = {start_vertex}
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex]]
    mst = [start_vertex]

    while edges:
        weight, frm, to = min(edges, key=lambda x: x[0])
        edges = [edge for edge in edges if edge[2] != to]
        if to not in visited:
            visited.add(to)
            mst.append(to)
            for neighbor, weight in graph[to]:
                if neighbor not in visited:
                    edges.append((weight, to, neighbor))
    return mst


def draw_path(
    screen,
    path: List[Tuple[int, int]],
    cell_size: int,
    current_index: int,
    max_length: int,
):
    """
    Draw the path on the screen using Pygame with a gradient neon glow effect.

    Args:
        screen: Pygame screen object.
        path (List[Tuple[int, int]]): The path to draw.
        cell_size (int): Size of each cell in the grid.
        current_index (int): Current index in the path for the animation.
        max_length (int): Maximum number of segments to display at once.
    """
    start = max(0, current_index - max_length)
    end = current_index + 1
    segments = path[start:end]
    colors = [
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

    for i, (x, y) in enumerate(segments):
        color = colors[i % len(colors)]  # Cycle through the color spectrum
        pygame.draw.rect(
            screen, color, (y * cell_size, x * cell_size, cell_size, cell_size)
        )


def main():
    pygame.init()
    grid_size = 32
    cell_size = 20
    screen = pygame.display.set_mode(
        (grid_size * cell_size, grid_size * cell_size), pygame.FULLSCREEN
    )
    pygame.display.set_caption("Dynamic Hamiltonian Cycle Visualization")

    grid = create_grid(grid_size)
    graph = convert_to_graph(grid)
    current_path = prim_mst(graph, (0, 0))
    path_index = 0
    path_length = 1000  # Number of path segments to display at once

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Clear screen with black background
        draw_path(screen, current_path, cell_size, path_index, path_length)
        pygame.display.flip()
        path_index += 3  # Increase speed by moving 3 indices per frame

        # Recalculate path when nearing the end
        if path_index >= len(current_path) - path_length // 2:
            new_start_vertex = current_path[-1]
            current_path = current_path[:path_index] + prim_mst(graph, new_start_vertex)

        clock.tick(30)  # Increase the framerate for smoother and faster animation

    pygame.quit()


if __name__ == "__main__":
    main()
