from networkx import neighbors
from Algorithm import Algorithm
import logging
from typing import List, Optional, Tuple, Any, Set
import heapq
import threading
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from Snake import Snake
from Utility import Node
from pygame.math import Vector2

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class A_STAR(Algorithm):
    """
    A* Algorithm implementation for pathfinding.

    This class extends the Algorithm base class and implements an advanced, optimized, and adaptive A* search algorithm,
    which is a popular and powerful pathfinding and graph traversal algorithm. It is particularly known for its effectiveness
    in finding the shortest path while maximizing the space between the snake and its neighbors, ensuring the game goes on
    for as long as possible. The algorithm utilizes various distance calculation functions and their inverses to calculate
    (asynchronously) all possible paths and select the optimum path that balances maximum distance and maximum space for the
    next moves while still reaching the goal. The implementation aims for perfection in all regards, ensuring efficiency,
    robustness, flexibility, and compatibility with the GameController.

    Attributes:
        grid (List[List[int]]): The grid representing the game environment.
    """

    def __init__(self, grid: List[List[Node]]) -> None:
        """
        Initialize the A* algorithm instance with a thread pool executor, ensuring meticulous management of threading and asynchronous operations.

        This constructor meticulously sets up a thread pool executor with a precisely defined number of worker threads, which is paramount for handling asynchronous tasks with high efficiency. The executor's role is critical for processing neighbor nodes in the pathfinding algorithm, ensuring that each node is evaluated with the utmost precision and speed, thereby optimizing the pathfinding capabilities of the A* algorithm.

        Attributes:
            grid (List[List[Node]]): The grid representing the game environment, structured as a list of lists containing Node objects.
            lock (threading.Lock): A lock to ensure thread safety during operations that modify shared resources.
            executor (concurrent.futures.ThreadPoolExecutor): A thread pool executor configured with a specific number of worker threads to handle concurrent tasks.
            explored_set (Set[Node]): A set to keep track of all the nodes that have been explored during the algorithm's execution to prevent reprocessing.
            frontier (List[Tuple[float, Node]]): A priority queue (min-heap) to manage the exploration front of the algorithm, prioritizing nodes based on their computed costs.

        Args:
            grid (List[List[Node]]): The grid that represents the environment in which the A* algorithm will operate, provided during the instantiation of the class.
        """
        super().__init__(grid)  # Initialize the base class with the grid.
        self.lock = (
            threading.Lock()
        )  # Initialize a threading lock to ensure thread safety.
        self.executor = ThreadPoolExecutor(
            max_workers=10
        )  # Set up the executor with 10 worker threads.
        self.grid = (
            grid  # Store the grid as an instance variable for use in pathfinding.
        )
        self.explored_set: Set[Node] = (
            set()
        )  # Initialize an empty set to track explored nodes.
        self.frontier: List[Tuple[float, Node]] = (
            []
        )  # Initialize an empty list to act as the frontier priority queue.
        self.path: List[Node] = []  # Initialize an empty list to store the path.

        # Log the initialization details with high verbosity for debugging and traceability.
        logging.debug(
            "A* Algorithm instance initialized with the following configuration:"
        )
        logging.debug(f"Grid size: {len(grid)}x{len(grid[0])} (rows x columns)")
        logging.debug(
            "Thread pool executor initialized with a maximum of 10 worker threads for handling asynchronous tasks in the pathfinding algorithm."
        )
        logging.debug(
            "Lock and thread-safe structures are in place to ensure the integrity of shared resources during concurrent operations."
        )

    def run_algorithm(self, snake: "Snake") -> Optional["Node"]:
        """
        Execute the advanced A* algorithm to find the optimal path for the snake to follow using asynchronous task processing.

        This method implements the core functionality of the advanced A* algorithm with asynchronous processing of neighbor nodes.
        It utilizes a thread pool executor to submit tasks for processing each neighbor node and collects the results to determine
        the optimal path. The method ensures that all resources are properly managed by shutting down the executor upon completion
        of the task processing.

        Args:
            snake (Snake): The snake object for which the path is being calculated.

        Returns:
            Optional[Node]: The next node in the optimal path for the snake to follow, or None if no path is found.
        """
        # Reinitialize the executor if it has been previously shut down
        if self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=10)
            logging.debug("Reinitialized thread pool executor with 10 workers.")

        # Initialize all necessary lists to their default states
        self.frontier: List[Tuple[int, "Node"]] = []
        heapq.heapify(self.frontier)
        self.explored_set: Set["Node"] = set()
        self.path: List["Node"] = []
        logging.debug(
            "Execution of the advanced A_STAR algorithm with asynchronous processing has commenced."
        )

        # Retrieve initial and goal states
        initial_state, goal_state = self.get_initstate_and_goalstate(snake)
        logging.debug(
            f"Initial state set to {initial_state}, goal state set to {goal_state}."
        )

        current_node = snake.head
        logging.debug(f"Current node (snake head) is set to {current_node}.")

        # Retrieve neighbors of the current node
        neighbors = self.get_neighbors(current_node, self.grid)
        logging.debug(f"Neighbors of the current node {current_node}: {neighbors}")

        # Initialize a list to hold future objects for asynchronous processing
        futures = []

        # Submit tasks for processing each neighbor
        for neighbor in neighbors:
            future = self.executor.submit(
                self.process_neighbor, snake, current_node, neighbor, goal_state
            )
            futures.append(future)
            logging.debug(
                f"Task for processing neighbor {neighbor} has been submitted to the executor."
            )

        # List to hold neighbors after processing
        neighbors_to_process = []

        # Collect results from futures as they complete
        for future in as_completed(futures):
            neighbor = future.result()
            if neighbor is not None:
                neighbors_to_process.append(neighbor)
                logging.debug(
                    f"Processed neighbor {neighbor} has been added to the list for further processing."
                )

        # Ensure the executor is not shut down prematurely
        if not self.executor._shutdown:
            self.executor.shutdown(wait=True)
            logging.debug(
                "Thread pool executor has been shut down after completing all tasks."
            )

        # Return the next node in the optimal path or None if no path is found
        next_node = self.select_next_node(neighbors_to_process)
        logging.debug(f"Next node in the optimal path: {next_node}")
        return next_node

    def close(self) -> None:
        """
        Method to terminate the thread pool executor, ensuring all threads are properly closed and resources are released.

        This method is a critical component of resource management within the A* algorithm implementation. It ensures that the thread pool executor,
        which is used extensively for parallel processing of nodes, is shut down gracefully. This shutdown process is essential to prevent any resource leaks,
        which could lead to performance degradation or system instability. By ensuring a clean and thorough shutdown, this method upholds the integrity and efficiency
        of the system's resource management.

        Raises:
            Exception: If the executor fails to shutdown properly, an exception is raised detailing the failure.
        """
        # Attempt to shutdown the executor and log the action
        try:
            # Command the executor to cease accepting new tasks and complete all existing tasks
            self.executor.shutdown(wait=True)
            # Log the successful shutdown
            logging.debug(
                "Thread pool executor has been shut down explicitly via the close method."
            )
            # Output to console for immediate user feedback
            print("Executor shutdown successfully.")
        except Exception as e:
            # Log the exception with detailed traceback
            logging.error(f"Failed to shutdown the executor: {e}", exc_info=True)
            # Raise the exception to signal the failure to higher-level management logic
            raise Exception(f"Executor shutdown failed: {e}") from e
        finally:
            # Additional cleanup actions can be performed here if necessary
            logging.info("Executor close method has completed execution.")

    def calculate_heuristic(self, goalstate: "Node", neighbor: "Node") -> float:
        """
        Calculate the heuristic value from the neighbor to the goal state using various distance functions and their inverses.

        This method calculates the heuristic value from the neighbor node to the goal state node using a combination of
        various distance calculation functions and their inverses. The distance functions used include Manhattan distance,
        Euclidean distance, Chebyshev distance, Minkowski distance, Octile distance, and their inverses. The method aims to
        find the most accurate and informative heuristic value that guides the algorithm towards the optimal path.

        Args:
            goalstate (Node): The goal state node.
            neighbor (Node): The neighbor node.

        Returns:
            float: The calculated heuristic value.
        """
        # Calculate the heuristic using Manhattan distance and its inverse
        manhattan_heuristic = self.manhattan_distance(goalstate, neighbor)
        inverse_manhattan_heuristic = 1 / (manhattan_heuristic + 1)

        # Calculate the heuristic using Euclidean distance and its inverse
        euclidean_heuristic = self.euclidean_distance(goalstate, neighbor)
        inverse_euclidean_heuristic = 1 / (euclidean_heuristic + 1)

        # Calculate the heuristic using Chebyshev distance and its inverse
        chebyshev_heuristic = self.chebyshev_distance(goalstate, neighbor)
        inverse_chebyshev_heuristic = 1 / (chebyshev_heuristic + 1)

        # Calculate the heuristic using Minkowski distance and its inverse
        minkowski_heuristic = self.minkowski_distance(goalstate, neighbor)
        inverse_minkowski_heuristic = 1 / (minkowski_heuristic + 1)

        # Calculate the heuristic using Octile distance and its inverse
        octile_heuristic = self.octile_distance(goalstate, neighbor)
        inverse_octile_heuristic = 1 / (octile_heuristic + 1)

        # Combine the heuristics using a weighted sum
        heuristic = (
            manhattan_heuristic * 0.2
            + inverse_manhattan_heuristic * 0.1
            + euclidean_heuristic * 0.2
            + inverse_euclidean_heuristic * 0.1
            + chebyshev_heuristic * 0.1
            + inverse_chebyshev_heuristic * 0.05
            + minkowski_heuristic * 0.1
            + inverse_minkowski_heuristic * 0.05
            + octile_heuristic * 0.1
            + inverse_octile_heuristic * 0.05
        )

        logging.debug(
            f"Heuristic value from {neighbor} to {goalstate} calculated as {heuristic} using a combination of distance functions and their inverses."
        )
        return heuristic

    async def get_neighbors_async(self, node: "Node") -> List["Node"]:
        """
        Asynchronously retrieves all neighbors and next-nearest neighbors (one extra move away) of the given node, ensuring all returned objects are instances of the Node class.

        This method is designed to operate asynchronously to ensure optimum efficiency in fetching neighbors. It leverages
        the thread pool executor to parallelize the neighbor retrieval process, thereby minimizing the response time and
        enhancing the performance of the A* algorithm. It also includes rigorous type checking to ensure that only Node instances are processed, preventing type mismatches that could lead to attribute errors.

        Args:
            node (Node): The node for which neighbors are to be retrieved.

        Returns:
            List[Node]: A list of neighbor nodes including the next-nearest neighbors, all confirmed to be instances of Node.
        """
        logging.debug(
            f"Initiating asynchronous retrieval of neighbors for node: {node}"
        )
        neighbors = []
        next_nearest_neighbors = []

        # Define the possible moves to find neighbors (up, down, left, right)
        directions = [
            Vector2(-1, 0),
            Vector2(1, 0),
            Vector2(0, -1),
            Vector2(0, 1),
        ]  # Up, Down, Left, Right

        # Submit tasks to executor to find direct neighbors
        future_neighbors = [
            self.executor.submit(self.find_neighbor, node, direction)
            for direction in directions
        ]

        # Collect results for direct neighbors
        for future in as_completed(future_neighbors):
            neighbor = future.result()
            if neighbor is not None and isinstance(neighbor, Node):
                neighbors.append(neighbor)
                logging.debug(f"Direct neighbor added: {neighbor}")

                # For each direct neighbor, find next-nearest neighbors
                future_next_neighbors = [
                    self.executor.submit(self.find_neighbor, neighbor, direction)
                    for direction in directions
                ]

                # Collect results for next-nearest neighbors
                for future_next in as_completed(future_next_neighbors):
                    next_neighbor = future_next.result()
                    if (
                        next_neighbor is not None
                        and next_neighbor not in neighbors
                        and isinstance(next_neighbor, Node)
                    ):
                        next_nearest_neighbors.append(next_neighbor)
                        logging.debug(f"Next-nearest neighbor added: {next_neighbor}")
            else:
                logging.error(
                    f"Expected Node, got {type(neighbor).__name__} from future result."
                )

        # Combine neighbors and next-nearest neighbors
        all_neighbors = neighbors + next_nearest_neighbors
        logging.debug(
            f"All neighbors retrieved asynchronously for node {node}: {all_neighbors}"
        )
        return all_neighbors

    def find_neighbor(self, node: "Node", direction: Vector2) -> Optional["Node"]:
        """
        This method is meticulously designed to identify and return a neighboring node based on a specified direction from a given node.
        It calculates the potential neighbor's coordinates by adding the directional vectors to the current node's coordinates.
        It then validates the computed position to determine if it represents a valid and accessible location within the game grid.
        If the position is valid and corresponds to a Node instance, the method logs this validation and returns the corresponding node.
        If not, it logs the invalidity of the position and returns None, indicating no valid neighbor in that direction.

        Args:
            node (Node): The node from which the neighbor is to be found. This node acts as the reference point for the calculation.
            direction (Vector2): The vector representing the direction in which the neighbor is to be searched. This vector is added to the current node's coordinates.

        Returns:
            Optional[Node]: The neighbor node if the position is valid and the node exists; otherwise, None. This ensures that only accessible nodes are considered as valid neighbors.
        """
        # Calculate the potential neighbor's coordinates by adding the direction vector to the current node's coordinates.
        potential_neighbor_coordinates = (
            int(node.x + direction.x),
            int(node.y + direction.y),
        )  # Explicit type conversion to int to ensure correct type handling.

        # Validate the coordinates are within the grid boundaries and the position corresponds to a Node instance.
        if 0 <= potential_neighbor_coordinates[0] < len(
            self.grid
        ) and 0 <= potential_neighbor_coordinates[1] < len(self.grid[0]):
            potential_neighbor = self.grid[potential_neighbor_coordinates[0]][
                potential_neighbor_coordinates[1]
            ]
            if isinstance(potential_neighbor, Node):
                # Log the successful validation of the neighbor's position.
                logging.debug(
                    f"Valid neighbor found at {potential_neighbor} from node {node} in direction {direction}"
                )
                # Return the valid neighbor node.
                return potential_neighbor
            else:
                # Log the type mismatch if the position does not correspond to a Node instance.
                logging.error(
                    f"Type mismatch: Expected Node, got {type(potential_neighbor).__name__} at position {potential_neighbor_coordinates}"
                )
        # Log the invalidity of the calculated position if it does not represent a valid grid location.
        logging.debug(
            f"Invalid neighbor position at coordinates {potential_neighbor_coordinates} from node {node} in direction {direction}"
        )
        # Return None to indicate that no valid neighbor was found in the specified direction.
        return None

    def process_neighbor(
        self, snake: "Snake", current_node: "Node", neighbor: "Node", goal_state: "Node"
    ) -> Optional["Node"]:
        """
        Process a neighbor node for the A* algorithm, ensuring it is valid and accessible.
        """
        # Check if the neighbor is a valid position
        if not self.valid_position(neighbor):
            logging.debug(f"Neighbor {neighbor} is an invalid position.")
            return None

        # Check if the neighbor is an obstacle
        if self.is_obstacle(neighbor):
            logging.debug(f"Neighbor {neighbor} is an obstacle.")
            return None

        # Calculate the cost to reach the neighbor from the current node
        self.calculate_cost(current_node, neighbor)

        # Calculate the heuristic value from the neighbor to the goal state
        neighbor_position = neighbor.x, neighbor.y
        neighbor.h = self.calculate_heuristic(goal_state, neighbor_position)

        # Calculate the total cost of the neighbor
        neighbor.f = neighbor.g + neighbor.h

        logging.debug(f"Neighbor {neighbor} processed successfully.")
        return neighbor

    def calculate_cost(self, current_node: "Node", neighbor: "Node") -> None:
        """
        Calculate the cost to reach the neighbor from the current node.
        """
        # Calculate the cost to reach the neighbor from the current node
        neighbor.g = current_node.g + 1
        logging.debug(
            f"Cost to reach neighbor {neighbor} from current node {current_node} calculated as {neighbor.g}"
        )

    def is_obstacle(self, node: "Node") -> bool:
        """
        Check if the node is an obstacle.
        """
        # Check if the node is an obstacle
        if node in self.explored_set:  # Check if the node is in the explored set
            logging.debug(f"Node {node} is an obstacle.")
            return True
        return False

    def valid_position(self, node: "Node") -> bool:
        """
        Check if the node is a valid position.
        """
        # Check if the node is a valid position
        if (
            node not in self.explored_set
        ):  # Check if the node is not in the explored set
            logging.debug(f"Node {node} is a valid position.")
            return True
        return False

    from networkx import neighbors

    def reconstruct_path(self, node: Node) -> List[Node]:
        """
        Reconstruct the path from the start node to the goal node.
        """
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]  # Return reversed path
