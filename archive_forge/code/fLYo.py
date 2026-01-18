from Algorithm import Algorithm
from typing import List, Optional, Type
import logging
import heapq
from collections import namedtuple

# Node definition using namedtuple for clarity and immutability
Node = namedtuple("Node", ["x", "y", "g", "h", "f", "parent"])

# Constants for clarity
STEP_COST = 1


# Configure logging within a function to avoid side effects when imported
def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    configure_logging()


class A_STAR(Algorithm):
    def __init__(self, grid: "Grid") -> None:
        super().__init__(grid)
        logging.debug("A_STAR algorithm initialized with grid.")

    def run_algorithm(self, snake: "Snake") -> Optional[List[Node]]:
        try:
            # Initialization of algorithm state
            self.frontier = []
            self.explored_set = []
            self.path = []

            initialstate: Node
            goalstate: Node
            initialstate, goalstate = self.get_initstate_and_goalstate(snake)

            # Using a priority queue for the frontier to optimize node selection
            heapq.heappush(self.frontier, (initialstate.f, initialstate))
            logging.debug(
                f"Initial state {initialstate} added to frontier with priority queue."
            )

            while self.frontier:
                # Efficient extraction of the node with the lowest f value
                lowest_f, lowest_node = heapq.heappop(self.frontier)
                logging.debug(f"Lowest node {lowest_node} popped from frontier.")

                if lowest_node == goalstate:
                    logging.info("Goal state reached.")
                    return self.get_path(lowest_node)

                self.explored_set.append(lowest_node)
                logging.debug(f"Node {lowest_node} added to explored set.")
                neighbors: List[Node] = self.get_neighbors(lowest_node)

                for neighbor in neighbors:
                    if (
                        self.inside_body(snake, neighbor)
                        or self.outside_boundary(neighbor)
                        or neighbor in self.explored_set
                    ):
                        logging.debug(
                            f"Skipping neighbor {neighbor} due to invalid conditions."
                        )
                        continue

                    g = lowest_node.g + STEP_COST
                    best = False

                    if neighbor not in [n for _, n in self.frontier]:
                        neighbor = neighbor._replace(
                            h=self.manhattan_distance(goalstate, neighbor)
                        )
                        heapq.heappush(self.frontier, (neighbor.f, neighbor))
                        best = True
                        logging.debug(
                            f"Neighbor {neighbor} added to frontier with updated heuristic."
                        )
                    else:
                        existing = next(
                            (n for _, n in self.frontier if n == neighbor), None
                        )
                        if existing and lowest_node.g < existing.g:
                            best = True

                    if best:
                        neighbor = neighbor._replace(
                            parent=lowest_node, g=g, f=g + neighbor.h
                        )
                        logging.debug(
                            f"Updated neighbor {neighbor} with new g, f values."
                        )
            logging.info("No path found to goal state.")
            return None
        except Exception as e:
            logging.error(
                f"An error occurred during the A* algorithm execution: {str(e)}"
            )
            return None
