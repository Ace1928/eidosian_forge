from Constants import NO_OF_CELLS
import logging

# Configure logging to the highest level of detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Node:
    def __init__(self, x: int, y: int):
        """
        Initialize a Node object with coordinates and default heuristic values.

        Parameters:
        x (int): The x-coordinate of the node.
        y (int): The y-coordinate of the node.
        """
        self.x: int = int(
            x
        )  # Explicit conversion to integer, ensuring type correctness
        self.y: int = int(
            y
        )  # Explicit conversion to integer, ensuring type correctness
        self.h: int = 0  # Heuristic cost to the goal, initialized to zero
        self.g: int = 0  # Cost from start to this node, initialized to zero
        self.f: int = (
            1000000  # Total cost, initialized to a high value to represent infinity
        )
        self.parent: "Node" = None  # Reference to the parent node, initialized to None
        logging.debug(
            f"Node created at position ({self.x}, {self.y}) with initial f value {self.f}"
        )

    def print_node(self) -> None:
        """
        Print the coordinates of the node.
        """
        logging.info(f"Node position - x: {self.x}, y: {self.y}")

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
