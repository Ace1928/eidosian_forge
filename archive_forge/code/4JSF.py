from Constants import NO_OF_CELLS
import logging

# Configure logging to the highest level of detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Node:
    def __init__(self, x: int, y: int, f: int = 0, g: int = 0, h: int = 0):
        """
        Initialize a Node object with coordinates and default heuristic values.

        Parameters:
        x (int): The x-coordinate of the node.
        y (int): The y-coordinate of the node.
        f (int): The total cost from start to goal through this node, initialized to zero.
        g (int): The cost from the start node to this node, initialized to zero.
        h (int): The heuristic cost from this node to the goal, initialized to zero.
        """
        self.x: int = int(
            x
        )  # Explicit conversion to integer, ensuring type correctness
        self.y: int = int(
            y
        )  # Explicit conversion to integer, ensuring type correctness
        self.f: int = int(
            f
        )  # Explicit conversion to integer, ensuring type correctness
        self.g: int = int(
            g
        )  # Explicit conversion to integer, ensuring type correctness
        self.h: int = int(
            h
        )  # Explicit conversion to integer, ensuring type correctness
        self.parent: "Node" = None  # Reference to the parent node, initialized to None
        logging.debug(
            f"Node created at position ({self.x}, {self.y}) with initial values f: {self.f}, g: {self.g}, h: {self.h}"
        )

    def __eq__(self, other: "Node") -> bool:
        """
        Equality comparison based on coordinates for identifying the same nodes.

        Parameters:
        other (Node): The other node to compare with.

        Returns:
        bool: True if this node's coordinates are the same as the other node's coordinates, False otherwise.
        """
        if not isinstance(other, Node):
            return NotImplemented
        equal_status: bool = (self.x, self.y) == (other.x, other.y)
        logging.debug(
            f"Node equality check: ({self.x}, {self.y}) == ({other.x}, {other.y}) -> {equal_status}"
        )
        return equal_status

    def __hash__(self) -> int:
        """
        Generate a hash based on the node's coordinates.

        Returns:
        int: The hash value of the node.
        """
        node_hash: int = hash((self.x, self.y))
        logging.debug(f"Hash generated for Node: {node_hash}")
        return node_hash

    def __lt__(self, other: "Node") -> bool:
        """
        Less than comparison based on f value for priority queue operations in pathfinding algorithms.

        Parameters:
        other (Node): The other node to compare with.

        Returns:
        bool: True if this node's f value is less than the other node's f value, False otherwise.
        """
        return self.f < other.f

    def __repr__(self) -> str:
        """
        Represent the Node object as a string for debugging and logging purposes.

        Returns:
        str: String representation of the Node.
        """
        node_representation: str = (
            f"Node({self.x}, {self.y}, f={self.f}, g={self.g}, h={self.h})"
        )
        logging.debug(f"Node representation: {node_representation}")
        return node_representation

    def print_node(self) -> None:
        """
        Print the coordinates and costs of the node.
        """
        logging.info(
            f"Node position - x: {self.x}, y: {self.y}, f: {self.f}, g: {self.g}, h: {self.h}"
        )

    def is_equal(self, other_node: "Node") -> bool:
        """
        Check if two nodes are at the same coordinates.

        Parameters:
        other_node (Node): The other node to compare with.

        Returns:
        bool: True if the nodes are at the same coordinates, False otherwise.
        """
        equal_status: bool = self.x == other_node.x and self.y == other_node.y
        logging.debug(
            f"Node equality check: ({self.x}, {self.y}) == ({other_node.x}, {other_node.y}) -> {equal_status}"
        )
        return equal_status


class Grid:
    def __init__(self):
        """
        Initialize a grid of nodes.
        """
        self.grid: list = []  # Initialize an empty list to hold the grid of nodes
        logging.debug("Initializing grid...")

        for i in range(NO_OF_CELLS):
            col: list = []  # Initialize an empty list to hold the column of nodes
            for j in range(NO_OF_CELLS):
                node: Node = Node(
                    i, j
                )  # Create a new Node object with coordinates (i, j)
                col.append(node)  # Append the newly created node to the column list
            self.grid.append(col)  # Append the filled column to the grid
            logging.debug(f"Column {i} added to grid with {len(col)} nodes.")

        logging.info(
            f"Grid initialized with {len(self.grid)} columns and {len(self.grid[0]) if self.grid else 0} rows per column."
        )
