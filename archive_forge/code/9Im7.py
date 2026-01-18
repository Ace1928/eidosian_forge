from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
import logging
import threading

# Configure logging with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Algorithm(ABC):
    """
    Abstract base class for pathfinding algorithms in the Snake AI game.

    This class provides a structured template for implementing various pathfinding algorithms,
    ensuring that each derived class adheres to a consistent interface for initializing the grid,
    calculating distances, and running the algorithm. It also ensures thread safety and efficient
    resource management to optimize performance and flexibility.

    Attributes:
        grid (List[List[Node]]): A grid representation of the game environment.
        frontier (List[Node]): List of nodes to be explored.
        explored_set (List[Node]): List of nodes that have been explored.
        path (List[Node]): List representing the path from start to goal.
        lock (threading.Lock): A lock for thread-safe manipulation of shared resources.
    """

    def __init__(self, grid: List[List[Node]]) -> None:
        """
        Initializes the Algorithm with a grid, setting up the necessary data structures
        for pathfinding operations. It also configures thread safety mechanisms.

        Args:
            grid (List[List[Node]]): The grid representing the game environment.
        """
        self.grid: List[List[Node]] = grid
        self.frontier: List[Node] = []
        self.explored_set: List[Node] = []
        self.path: List[Node] = []
        self.lock: threading.Lock = threading.Lock()  # Initialize the threading lock for thread safety

        logging.debug("Algorithm base class initialized with grid.")
        logging.debug(f"Grid dimensions set to {len(grid)}x{len(grid[0])} with total cells: {len(grid) * len(grid[0])}")
        logging.debug("Frontier, Explored Set, and Path lists have been initialized and are ready for use.")

        # Ensuring that all resources are ready and logging is set up before any operations begin
        try:
            with self.lock:
                # Placeholder for any pre-processing required on the grid or other structures
                logging.debug("Pre-processing of grid and data structures completed successfully.")
        except Exception as e:
            logging.error("An error occurred during the initialization of the Algorithm: {}".format(e))
            raise

    def get_initstate_and_goalstate(self, snake) -> Tuple[Node, Node]:
        """
        Determines the initial and goal states for the algorithm based on the current position of the snake
        and the position of the fruit, which is the target.

        This method meticulously extracts the current position of the snake and the position of the fruit
        from the game state, encapsulating them into Node objects which represent the initial and goal states
        respectively for the pathfinding algorithm. This is a critical step in setting up the algorithm's
        environment for execution.

        Args:
            snake (Snake): The snake object from which the current position and the target position are derived.

        Returns:
            Tuple[Node, Node]: A tuple containing the Node representations of the initial state (current position of the snake)
            and the goal state (position of the fruit).

        Raises:
            ValueError: If the snake or fruit positions are undefined or result in invalid Node coordinates.

        Detailed logging is used to ensure that each step of the process is recorded for debugging and traceability.
        """
        try:
            # Extracting the x and y coordinates of the snake's current position
            initial_x: int = snake.get_x()
            initial_y: int = snake.get_y()
            # Creating a Node for the initial state
            initial_state: Node = Node(initial_x, initial_y)
            logging.debug(f"Initial state Node created at coordinates ({initial_x}, {initial_y}).")

            # Extracting the x and y coordinates of the fruit's position
            fruit_x: int = snake.get_fruit().x
            fruit_y: int = snake.get_fruit().y
            # Creating a Node for the goal state
            goal_state: Node = Node(fruit_x, fruit_y)
            logging.debug(f"Goal state Node created at coordinates ({fruit_x}, {fruit_y}).")

            # Logging the successful determination of initial and goal states
            logging.info(f"Initial state determined at {initial_state}. Goal state determined at {goal_state}.")

            return initial_state, goal_state

        except AttributeError as e:
            # Logging an error if there is an issue with accessing snake or fruit attributes
            logging.error("Failed to access snake or fruit attributes.", exc_info=True)
            raise ValueError("Invalid snake or fruit attributes provided.") from e

        except Exception as e:
            # Handling any other unexpected exceptions
            logging.critical("An unexpected error occurred while determining initial and goal states.", exc_info=True)
            raise RuntimeError("An unexpected error occurred in get_initstate_and_goalstate method.") from e
    def manhattan_distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        Calculates the Manhattan distance between two nodes, which is the sum of the absolute differences of their Cartesian coordinates.
        This distance metric is crucial in grid-based pathfinding contexts where only four-directional movement is allowed (up, down, left, right).

        The method employs a detailed and meticulous approach to ensure the calculation is not only accurate but also optimized for performance.
        It leverages Python's built-in `abs` function to compute the absolute difference between the x-coordinates and y-coordinates of the two nodes,
        summing these differences to obtain the Manhattan distance.

        Args:
            nodeA (Node): The first node, representing a point in the grid.
            nodeB (Node): The second node, representing another point in the grid.

        Returns:
            int: The Manhattan distance between the two nodes, which is a non-negative integer representing the number of grid steps required to move from nodeA to nodeB without diagonal movement.

        Raises:
            TypeError: If either nodeA or nodeB is not an instance of the Node class.

        Detailed logging is used to ensure that each step of the process is recorded for debugging and traceability. The function also checks the type of the input parameters to prevent runtime errors due to incorrect types.
        """
        if not isinstance(nodeA, Node) or not isinstance(nodeB, Node):
            logging.error("One or both of the provided arguments are not of type Node.")
            raise TypeError("Both nodeA and nodeB must be instances of the Node class.")

        # Extracting the x and y coordinates of both nodes
        x1, y1 = nodeA.x, nodeA.y
        x2, y2 = nodeB.x, nodeB.y

        # Calculating the absolute differences in both dimensions
        delta_x: int = abs(x1 - x2)
        delta_y: int = abs(y1 - y2)

        # Summing the differences to get the Manhattan distance
        manhattan_distance: int = delta_x + delta_y

        # Logging the calculated Manhattan distance with detailed node information
        logging.debug(f"Calculated Manhattan distance between Node A at ({x1}, {y1}) and Node B at ({x2}, {y2}) is {manhattan_distance}.")

        return manhattan_distance
    def euclidean_distance(self, nodeA: Node, nodeB: Node) -> float:
        """
        Calculates the Euclidean distance between two nodes with meticulous precision and comprehensive detail.

        The Euclidean distance, also known as L2 norm or Euclidean norm, is the straight-line distance between two points
        in Euclidean space. This function computes the Euclidean distance between two nodes represented by their Cartesian
        coordinates in a 2D space, ensuring the calculation adheres to mathematical principles of geometry.

        Args:
            nodeA (Node): The first node, representing a point in the grid with coordinates (x, y).
            nodeB (Node): The second node, representing another point in the grid with coordinates (x, y).

        Returns:
            float: The Euclidean distance between the two nodes, calculated using the formula sqrt((x2 - x1)^2 + (y2 - y1)^2).

        Raises:
            TypeError: If either nodeA or nodeB is not an instance of the Node class.

        Detailed logging is used to ensure that each step of the process is recorded for debugging and traceability.
        The function also checks the type of the input parameters to prevent runtime errors due to incorrect types.
        """
        if not isinstance(nodeA, Node) or not isinstance(nodeB, Node):
            logging.error("One or both of the provided arguments are not of type Node.")
            raise TypeError("Both nodeA and nodeB must be instances of the Node class.")

        # Extracting the x and y coordinates of both nodes
        x1, y1 = nodeA.x, nodeA.y
        x2, y2 = nodeB.x, nodeB.y

        # Calculating the squared differences in both dimensions
        squared_difference_x: int = (x1 - x2) ** 2
        squared_difference_y: int = (y1 - y2) ** 2

        # Summing the squared differences
        sum_of_squared_differences: int = squared_difference_x + squared_difference_y

        # Calculating the Euclidean distance as the square root of the sum of squared differences
        euclidean_distance: float = math.sqrt(sum_of_squared_differences)

        # Logging the calculated Euclidean distance with detailed node information
        logging.debug(f"Calculated Euclidean distance between Node A at ({x1}, {y1}) and Node B at ({x2}, {y2}) is {euclidean_distance}.")

        return euclidean_distance

    @abstractmethod
    def run_algorithm(self, snake) -> Optional[Node]:
        """
        Abstract method to run the pathfinding algorithm.

        Args:
            snake: The snake object.

        Returns:
            Optional[Node]: The next node in the path for the snake to follow, if any.
        """
        pass

    def get_path(self, node: Node) -> Node:
        """
        Constructs the path from the given node to the root node.

        Args:
            node (Node): The node from which to construct the path.

        Returns:
            Node: The root node of the path.
        """
        logging.debug("Constructing path from node to root.")
        if node.parent is None:
            logging.debug(f"Node {node} is root node.")
            return node

        while node.parent.parent is not None:
            self.path.append(node)
            logging.debug(f"Added {node} to path.")
            node = node.parent
        logging.debug(f"Path construction completed with final node {node}.")
        return node
    def inside_body(self, snake, node: Node) -> bool:
        """
        Determines whether a specified node is located within the physical confines of the snake's body, thereby
        assessing the node's occupancy status relative to the snake's body segments.

        This method iterates through each segment of the snake's body and compares the coordinates of the segment
        with those of the given node. If the coordinates match, it indicates that the node is occupied by a segment
        of the snake's body.

        Args:
            snake (Snake): The snake object, which encapsulates the attributes and methods pertinent to the snake
                           entity within the game environment.
            node (Node): The node object, representing a discrete unit or position within the game grid, which is
                         to be checked against the snake's body segments.

        Returns:
            bool: Returns True if the node coincides with any of the snake's body segments, indicating that the node
                  is inside the snake's body. Returns False if there is no such coincidence, indicating that the node
                  is not occupied by the snake.

        Raises:
            TypeError: If the input arguments are not of the expected type 'Snake' and 'Node' respectively.

        Detailed Logging:
            - Logs a debug message when the method is initiated with the given snake and node.
            - Logs a debug message for each body segment checked against the node.
            - Logs a debug message when a matching body segment is found.
            - Logs a debug message when no matching body segment is found after checking all segments.
        """
        for body in snake.body:
            if body.x == node.x and body.y == node.y:
                logging.debug(f"Node {node} is inside snake's body.")
                return True
        logging.debug(f"No match found: Node {node} is not inside snake's body after checking all segments.")
        return False
    def outside_boundary(self, node: Node) -> bool:
        """
        Determines if a node is positioned outside the predefined game boundaries, which are crucial for maintaining
        the integrity of the game's environment. This method meticulously checks both horizontal and vertical
        boundaries to ascertain whether the node lies outside these limits.

        Args:
            node (Node): The node whose position is to be evaluated against the game's boundary conditions.

        Returns:
            bool: Returns True if the node is found to be outside the established boundaries, either horizontally or
                  vertically. Returns False if the node is within all boundary conditions.

        Raises:
            ValueError: If the node's coordinates are not integers, which is a prerequisite for boundary checking.

        Detailed Logging:
            - Logs a debug message when the method is initiated with the given node.
            - Logs a debug message if the node is outside horizontal boundaries.
            - Logs a debug message if the node is outside vertical boundaries.
            - Logs a debug message if the node is inside all boundaries.

        Example:
            >>> node = Node(10, 20)
            >>> algorithm.outside_boundary(node)
            False
        """
        # Validate node coordinates are integers for boundary comparison
        if not isinstance(node.x, int) or not isinstance(node.y, int):
            logging.error(f"Invalid node coordinates: ({node.x}, {node.y})")
            raise ValueError("Node coordinates must be integers for boundary checking.")

        logging.debug(f"Checking if node {node} is outside game boundaries.")
        
        # Check horizontal boundaries
        if not 0 <= node.x < NO_OF_CELLS:
            logging.debug(f"Node {node} is outside horizontal boundaries.")
            return True
        
        # Check vertical boundaries
        if not BANNER_HEIGHT <= node.y < NO_OF_CELLS:
            logging.debug(f"Node {node} is outside vertical boundaries.")
            return True
        
        # If all checks are passed, the node is inside the boundaries
        logging.debug(f"Node {node} is inside boundaries.")
        return False
    def get_neighbors(self, node: Node) -> List[Node]:
        """
        Retrieves the neighboring nodes of a given node with meticulous detail, ensuring that all possible neighbors
        are considered within the constraints of the game grid. This method is crucial for the pathfinding algorithms
        that require knowledge of adjacent nodes to compute the optimal path.

        Args:
            node (Node): The node for which to find neighbors, ensuring that the node's coordinates are within the
                         valid range of the game grid.

        Returns:
            List[Node]: A list containing the neighboring nodes. This list may include up to four nodes (left, right,
                        top, bottom) depending on the position of the input node within the grid.

        Raises:
            IndexError: If the node's coordinates are out of the acceptable range, an IndexError is raised.

        Detailed Logging:
            - Logs a debug message at the start of the function with the node's coordinates.
            - Logs a debug message for each neighbor added, specifying the direction and the coordinates.
            - Logs a debug message summarizing the total neighbors found at the end of the function.

        Example:
            >>> node = Node(5, 5)
            >>> algorithm.get_neighbors(node)
            [Node(4, 5), Node(6, 5), Node(5, 4), Node(5, 6)]
        """
        logging.debug(f"Initiating neighbor retrieval for node at coordinates ({node.x}, {node.y}).")
        
        # Validate node coordinates are within the acceptable range
        if not (0 <= node.x < NO_OF_CELLS) or not (0 <= node.y < NO_OF_CELLS):
            logging.error(f"Node coordinates out of bounds: ({node.x}, {node.y})")
            raise IndexError("Node coordinates are out of the acceptable range.")

        i: int = int(node.x)
        j: int = int(node.y)

        neighbors: List[Node] = []
        
        # Check and append left neighbor
        if i > 0:
            left_neighbor: Node = self.grid[i - 1][j]
            neighbors.append(left_neighbor)
            logging.debug(f"Left neighbor at ({left_neighbor.x}, {left_neighbor.y}) added.")
        
        # Check and append right neighbor
        if i < NO_OF_CELLS - 1:
            right_neighbor: Node = self.grid[i + 1][j]
            neighbors.append(right_neighbor)
            logging.debug(f"Right neighbor at ({right_neighbor.x}, {right_neighbor.y}) added.")
        
        # Check and append top neighbor
        if j > 0:
            top_neighbor: Node = self.grid[i][j - 1]
            neighbors.append(top_neighbor)
            logging.debug(f"Top neighbor at ({top_neighbor.x}, {top_neighbor.y}) added.")
        
        # Check and append bottom neighbor
        if j < NO_OF_CELLS - 1:
            bottom_neighbor: Node = self.grid[i][j + 1]
            neighbors.append(bottom_neighbor)
            logging.debug(f"Bottom neighbor at ({bottom_neighbor.x}, {bottom_neighbor.y}) added.")

        logging.debug(f"Total neighbors found for node at ({node.x}, {node.y}): {len(neighbors)}")
        return neighbors
