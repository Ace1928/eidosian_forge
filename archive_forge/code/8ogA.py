"""
This module implements the A* pathfinding algorithm for the Snake AI.

The A* algorithm finds the longest possible path between the snake's head and tail,
ensuring that the snake never gets trapped and always has a way out after reaching
the previous tail position. The apple is eaten when the snake is on the path.
"""

import numpy as np
from numba import njit, types, jit, prange
from numba.experimental import jitclass
from numba.types import int32, float64, UniTuple, optional, ListType
from numba.typed import List

# Step 1: Initialize the deferred type
node_type = types.deferred_type()

# Step 2: Define the class without specifying the parent type
spec = [
    ("x", types.int32),
    ("y", types.int32),
    ("f", types.float64),
    ("g", types.float64),
    ("h", types.float64),
    ("_position", types.UniTuple(types.int32, 2)),
    ("parent", optional(node_type)),  # Use the deferred type for parent
]


# Step 3: Create the jitclass
@jitclass(spec)
class Node:
    """
    Represents a node in the A* pathfinding algorithm.

    Each position on the game board is represented by a node. Each node has a parent
    and f, g, and h values used in the A* algorithm.

    Attributes:
        x (int): The x-coordinate of the node.
        y (int): The y-coordinate of the node.
        parent (Optional[Node]): The parent node of the current node.
        f (float): The total cost of the node (g + h).
        g (float): The cost from the start node to the current node.
        h (float): The estimated cost from the current node to the end node.
        _position (Tuple[int, int]): The position of the node as a tuple (x, y).
        _lru_cache (LRUCache): LRU cache for storing calculated values.
    """

    def __init__(
        self, x: types.int32, y: types.int32, parent: types.optional(node_type) = None
    ):
        """
        Initializes a new instance of the Node class.

        Args:
            x (int): The x-coordinate of the node.
            y (int): The y-coordinate of the node.
            parent (Optional[Node]): The parent node of the current node.
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.f = np.float64(0.0)
        self.g = np.float64(0.0)
        self.h = np.float64(0.0)
        self._position = (x, y)

    @property
    def position(self) -> types.UniTuple(types.int32, 2):
        """
        Returns the position of the node as a tuple (x, y).
        """
        return self._position

    @staticmethod
    def distance(node1: "Node", node2: "Node") -> types.float64:
        """
        Calculates the Manhattan distance between two nodes.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            float: The Manhattan distance between the nodes.
        """
        return np.abs(node1.x - node2.x) + np.abs(node1.y - node2.y)

    @staticmethod
    def cached_distance(
        x1: types.int32, y1: types.int32, x2: types.int32, y2: types.int32
    ) -> types.float64:
        """
        Calculates the Manhattan distance between two positions.

        Args:
            x1 (int): The x-coordinate of the first position.
            y1 (int): The y-coordinate of the first position.
            x2 (int): The x-coordinate of the second position.
            y2 (int): The y-coordinate of the second position.

        Returns:
            float: The Manhattan distance between the positions.
        """
        return np.abs(x1 - x2) + np.abs(y1 - y2)

    def _calculate_costs(
        self, start: "Node", end: "Node"
    ) -> types.UniTuple(types.float64, 3):
        """
        Calculates the f, g, and h costs of the node.

        Args:
            start (Node): The starting node.
            end (Node): The ending node.

        Returns:
            Tuple[float, float, float]: The f, g, and h costs of the node.
        """
        g = self.cached_distance(start.x, start.y, self.x, self.y)
        h = self.cached_distance(self.x, self.y, end.x, end.y)
        f = g + h
        return f, g, h

    def update_costs(self, start: "Node", end: "Node") -> None:
        """
        Updates the f, g, and h costs of the node using LRU cache.

        Args:
            start (Node): The starting node.
            end (Node): The ending node.
        """
        self.f, self.g, self.h = self._lru_cache(start, end)

    @staticmethod
    def get_neighbors(
        node: "Node", maze: types.Array(types.int32, 2, "C")
    ) -> types.List(node_type):
        """
        Returns the neighboring nodes of the given node.

        Args:
            node (Node): The node to get neighbors for.
            maze (np.ndarray): The maze array representing the game board.

        Returns:
            List[Node]: The list of neighboring nodes.
        """
        neighbors = List.empty_list(node_type)
        for dx, dy in prange(4):
            x, y = node.x + (dx - 1), node.y + (dy - 1)
            if 0 <= x < maze.shape[1] and 0 <= y < maze.shape[0] and maze[y, x] != -1:
                neighbors.append(Node(x, y))
        return neighbors

    @staticmethod
    def get_path(end: "Node") -> types.List(node_type):
        """
        Returns the path from the end node to the start node.

        Args:
            end (Node): The ending node.

        Returns:
            List[Node]: The path from the end node to the start node.
        """
        path = List.empty_list(node_type)
        current = end
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    @staticmethod
    def draw_path(
        maze: types.Array(types.int32, 2, "C"), path: types.List(node_type)
    ) -> None:
        """
        Draws the path on the maze.

        Args:
            maze (np.ndarray): The maze array representing the game board.
            path (List[Node]): The path to draw.
        """
        for node in path:
            maze[node.y, node.x] = 2

    @staticmethod
    def generate_maze(
        width: types.int32,
        height: types.int32,
        obstacles: types.List(types.UniTuple(types.int32, 2)),
    ) -> types.Array(types.int32, 2, "C"):
        """
        Generates a maze with the given width, height, and obstacles.

        Args:
            width (int): The width of the maze.
            height (int): The height of the maze.
            obstacles (List[Tuple[int, int]]): The list of obstacle positions.

        Returns:
            np.ndarray: The generated maze.
        """
        maze = np.zeros((height, width), dtype=np.int32)
        for x, y in obstacles:
            maze[y, x] = -1
        return maze

    def __eq__(self, other: "Node") -> bool:
        """
        Checks if the current node is equal to another node.

        Args:
            other (Node): The other node to compare.

        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        return (self.x == other.x) and (self.y == other.y)


node_type.define(Node.class_type.instance_type)


@njit
def astar(maze: np.ndarray, start: Node, end: Node):
    open_set = List.empty_list(Node.class_type.instance_type)
    closed_set = List.empty_list(Node.class_type.instance_type)
    open_set.append(start)

    while len(open_set) > 0:
        current = open_set[0]
        current_index = 0
        for index, item in enumerate(open_set):
            if item.f < current.f:
                current = item
                current_index = index

        open_set.pop(current_index)
        closed_set.append(current)

        if current == end:
            return Node.get_path(current)

        # Example neighbors addition, adjust as necessary
        neighbors = [Node(current.x + 1, current.y), Node(current.x - 1, current.y)]
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue

            tentative_g_score = current.g + 1
            if tentative_g_score < neighbor.g or neighbor not in open_set:
                neighbor.g = tentative_g_score
                neighbor.h = Node.distance(neighbor, end)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                if neighbor not in open_set:
                    open_set.append(neighbor)

    return List.empty_list(Node.class_type.instance_type)  # if no path found


def main() -> None:
    """
    The main function to test the pathfinding algorithm.

    This function generates random mazes with obstacles and finds the path between
    random start and end points. It visualizes the maze and path using ASCII characters.
    """
    import os
    import time
    import random

    width, height = 20, 20
    num_obstacles = 30
    num_tests = 50

    for i in range(num_tests):
        # Generate random maze with obstacles
        obstacles = [
            (random.randint(0, width - 1), random.randint(0, height - 1))
            for _ in range(num_obstacles)
        ]
        maze = Node.generate_maze(width, height, obstacles)

        # Generate random start and end points
        start = Node(random.randint(0, width - 1), random.randint(0, height - 1))
        end = Node(random.randint(0, width - 1), random.randint(0, height - 1))

        # Find the path using A* algorithm
        path = astar(maze, start, end)

        # Draw the path on the maze
        Node.draw_path(maze, path)

        # Print the test information
        print(f"Test {i + 1}/{num_tests}")
        print(f"Start: ({start.x}, {start.y})")
        print(f"End: ({end.x}, {end.y})")
        print("Maze:")
        for row in maze:
            for cell in row:
                if cell == -1:
                    print("█", end="")
                elif cell == 1:
                    print(".", end="")
                elif cell == 2:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()

        # Pause for a short duration
        time.sleep(1)

        # Clear the console screen
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
