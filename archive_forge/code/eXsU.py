import pygame
import random
from typing import List, Tuple, Dict, Set


def create_grid(size: int) -> List[List[int]]:
    """
    Methodically constructs a two-dimensional square grid of specified size where each cell within the grid is initialized to zero. This grid is represented as a list of lists, where each inner list corresponds to a row in the grid.

    Args:
        size (int): The dimension of the grid, specifically indicating both the number of rows and the number of columns, as the grid is square.

    Returns:
        List[List[int]]: A meticulously constructed 2D list where each element within the sublist (representing a row) is initialized to zero. The outer list encapsulates all these rows, thus forming the complete grid structure.

    Detailed Explanation:
        - The function begins by initializing an empty list named 'grid', which will eventually contain all the rows of the grid.
        - A loop iterates 'size' times, each iteration corresponding to the creation of a row in the grid. Within each iteration of this loop:
            - An empty list named 'row' is initialized to serve as the current row being constructed.
            - Another loop iterates 'size' times to populate the 'row' list with zeros. Each iteration appends a zero to the 'row' list, representing an individual cell in the grid initialized to zero.
            - After constructing the complete 'row' filled with zeros, this 'row' is appended to the 'grid' list.
        - Once all rows have been constructed and appended to the 'grid', the fully constructed grid is returned.
    """
    grid = []  # Initialize the outer list that will hold all rows of the grid.
    for row_index in range(size):  # Loop to create each row in the grid.
        row = []  # Initialize the current row list that will be filled with zeros.
        for column_index in range(size):  # Loop to fill the current row with zeros.
            row.append(
                0
            )  # Append a zero to the current row list, representing an initialized cell.
        grid.append(row)  # Append the fully constructed row to the grid list.
    return grid  # Return the fully constructed grid, a list of lists filled with zeros.


def convert_to_graph(
    grid: List[List[int]],
) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
    """
    This function meticulously converts a two-dimensional grid into a graph representation where each cell in the grid is treated as a node in the graph. The nodes are connected to their adjacent nodes (up, down, left, right) with edges that have randomly assigned weights. The purpose of this conversion is to facilitate the representation of the grid in a format that is amenable to graph-theoretic algorithms.

    Args:
        grid (List[List[int]]): A two-dimensional list of integers representing the grid. Each element in the grid is assumed to be an integer.

    Returns:
        Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]: A dictionary representing the graph. The keys are tuples representing the coordinates of each node in the grid. The values are lists of tuples, where each tuple contains a tuple representing the coordinates of an adjacent node and a float representing the weight of the edge connecting the node to the adjacent node.

    Detailed Explanation:
        - The function initializes an empty dictionary named 'graph' to store the graph representation.
        - The size of the grid is determined by measuring the length of the grid list, which corresponds to the number of rows in the grid.
        - Two nested loops iterate over each cell in the grid. The outer loop variable 'i' represents the row index, and the inner loop variable 'j' represents the column index.
        - For each cell at position (i, j), an empty list named 'connections' is initialized to store the edges connecting the node at (i, j) to its adjacent nodes.
        - Conditional checks are performed to determine if adjacent nodes exist (i.e., the node is not on the boundary of the grid). For each existing adjacent node, a tuple containing the coordinates of the adjacent node and a randomly generated weight (using random.random()) is appended to the 'connections' list.
        - The node at (i, j) along with its 'connections' list is then added to the 'graph' dictionary.
        - After all nodes and their connections have been processed, the 'graph' dictionary is returned, providing a complete graph representation of the input grid.
    """
    size = len(grid)  # Determine the size of the grid based on the length of the list
    graph = {}  # Initialize an empty dictionary to store the graph representation

    # Iterate over each cell in the grid to construct the graph
    for i in range(size):
        for j in range(size):
            connections = (
                []
            )  # Initialize an empty list to store connections for the current node

            # Check for adjacent nodes and add connections with random weights
            if i < size - 1:  # Check if the node below exists
                connections.append(
                    ((i + 1, j), random.random())
                )  # Add connection to the node below with a random weight
            if j < size - 1:  # Check if the node to the right exists
                connections.append(
                    ((i, j + 1), random.random())
                )  # Add connection to the node to the right with a random weight
            if i > 0:  # Check if the node above exists
                connections.append(
                    ((i - 1, j), random.random())
                )  # Add connection to the node above with a random weight
            if j > 0:  # Check if the node to the left exists
                connections.append(
                    ((i, j - 1), random.random())
                )  # Add connection to the node to the left with a random weight

            # Add the current node and its connections to the graph dictionary
            graph[(i, j)] = connections

    return graph  # Return the fully constructed graph


def prim_mst(
    graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start_vertex: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    This function implements Prim's algorithm to compute the Minimum Spanning Tree (MST) from a given graph starting from a specified vertex. Prim's algorithm is a greedy algorithm that finds a minimum spanning tree for a weighted undirected graph. This means it finds a subset of the edges that forms a tree that includes every vertex, where the total weight of all the edges in the tree is minimized.

    Args:
        graph (Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]): This is a dictionary representing the graph. The keys are tuples representing the coordinates of each node in the grid, and the values are lists of tuples, where each tuple contains a tuple representing the coordinates of an adjacent node and a float representing the weight of the edge connecting the node to the adjacent node.
        start_vertex (Tuple[int, int]): This tuple represents the starting vertex from which the MST will begin to be computed.

    Returns:
        List[Tuple[int, int]]: This list contains tuples representing the nodes that form the Minimum Spanning Tree, starting from the start_vertex and expanding out to include all reachable nodes in the graph in a way that minimizes the total edge weight.
    """
    # Initialize a set to keep track of visited vertices to avoid revisiting and to ensure the MST does not contain cycles.
    visited = set([start_vertex])
    # Initialize a list to keep track of the edges that are being considered for addition to the MST. Start with all edges emanating from the start_vertex.
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex]]
    # Initialize the MST with the start_vertex since that's where the algorithm starts.
    mst = [start_vertex]

    # Continue looping as long as there are edges that might be added to the MST.
    while edges:
        # Select the edge with the minimum weight that does not form a cycle.
        weight, frm, to = min(edges, key=lambda x: x[0])
        # Remove all edges going to the 'to' vertex to prevent cycles.
        edges = [edge for edge in edges if edge[2] != to]
        # If 'to' vertex has not been visited, it is added to the MST.
        if to not in visited:
            # Mark this vertex as visited.
            visited.add(to)
            # Add this vertex to the MST.
            mst.append(to)
            # Consider all adjacent vertices for inclusion in the MST.
            for neighbor, weight in graph[to]:
                # Only consider vertices that have not been visited to prevent cycles.
                if neighbor not in visited:
                    # Add the edge to the list of potential edges to be added to the MST.
                    edges.append((weight, to, neighbor))
    # Return the list of vertices that form the MST.
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
    grid_size = 200
    cell_size = 5
    screen = pygame.display.set_mode(
        (grid_size * cell_size, grid_size * cell_size), pygame.RESIZABLE
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
        path_index += 1  # Increase speed by moving 3 indices per frame

        # Recalculate path when nearing the end
        if path_index >= len(current_path) - path_length // 2:
            new_start_vertex = current_path[-1]
            current_path = current_path[:path_index] + prim_mst(graph, new_start_vertex)

        clock.tick(30)  # Increase the framerate for smoother and faster animation

    pygame.quit()


if __name__ == "__main__":
    main()
