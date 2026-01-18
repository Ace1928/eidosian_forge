# Import required modules
from typing import (
    List,
    Optional,
    Tuple,
    Set,
    Dict,
)  # Import specific types from typing module
import pygame  # Import pygame module for game development
from pygame.math import (
    Vector2,
)  # Import Vector2 class from pygame.math module for vector calculations
from random import (
    randint,
)  # Import randint function from random module for generating random numbers
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality

# Initialize Pygame and its display module
pygame.init()
pygame.display.init()

# Retrieve the current display information
display_info = pygame.display.Info()


def calculate_block_size(screen_width: int, screen_height: int) -> int:
    """
    Calculate the block size based on the screen resolution.

    This function calculates the block size dynamically based on the screen resolution
    to ensure visibility and proportionality. It takes the screen width and height as
    input and returns the calculated block size as an integer.

    Args:
        screen_width (int): The width of the screen in pixels.
        screen_height (int): The height of the screen in pixels.

    Returns:
        int: The calculated block size.
    """
    # Define the reference resolution and corresponding block size
    reference_resolution = Vector2(1920, 1080)
    reference_block_size = 20

    # Calculate the scaling factor based on the reference resolution
    scaling_factor = min(
        screen_width / reference_resolution.x, screen_height / reference_resolution.y
    )

    # Calculate the block size dynamically based on the screen size
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))

    # Ensure the block size does not become too large or too small
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


# Apply the calculated block size based on the current screen resolution
BLOCK_SIZE = calculate_block_size(display_info.current_w, display_info.current_h)

# Define the border width as equivalent to 3 blocks
BORDER_WIDTH = 3 * BLOCK_SIZE  # Width of the border to be subtracted from each side

# Define the screen size with a proportional border around the edges
SCREEN_SIZE = (
    display_info.current_w - 2 * BORDER_WIDTH,
    display_info.current_h - 2 * BORDER_WIDTH,
)

# Define a constant for the border color as solid white
BORDER_COLOR = (255, 255, 255)  # RGB color code for white

# Instantiate the Clock object for controlling the game's frame rate
CLOCK = pygame.time.Clock()

# Define the desired frames per second
FRAMES_PER_SECOND = 60

# Calculate the tick rate based on the desired FPS
TICK_RATE = 1000 // FRAMES_PER_SECOND


def setup() -> Tuple[pygame.Surface, "Pathfinder", pygame.time.Clock]:
    """
    Initialize the game environment, setting up the display and instantiating game objects.

    This function initializes Pygame, sets up the screen surface, instantiates the Pathfinder
    object, initiates the pathfinding algorithm, and returns the necessary objects for the game.

    Returns:
        Tuple[pygame.Surface, Pathfinder, pygame.time.Clock]:
            - screen (pygame.Surface): The game screen surface.
            - search_object (Pathfinder): The Pathfinder object for pathfinding.
            - clock_object (pygame.time.Clock): The clock object for controlling the game's frame rate.
    """
    # Initialize Pygame
    pygame.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pygame.Surface = pygame.display.set_mode(SCREEN_SIZE)
    # Instantiate the Pathfinder object with screen size and logger
    search_object = Pathfinder(SCREEN_SIZE[0], SCREEN_SIZE[1], logging.getLogger())
    # Initiate the pathfinding algorithm
    search_object.get_path()
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock_object: pygame.time.Clock = CLOCK
    return screen, search_object, clock_object


class Pathfinder:
    """
    A class dedicated to pathfinding within a grid-based environment using the A* algorithm.

    Attributes:
        grid_width (int): The width of the grid in which pathfinding is performed.
        grid_height (int): The height of the grid in which pathfinding is performed.
        logger (logging.Logger): Logger for recording operational logs.
        obstacles (Set[pygame.math.Vector2]): A set of obstacles within the grid, represented by their positions.
    """

    def __init__(
        self,
        grid_width: int = 100,
        grid_height: int = 100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the Pathfinder object with the necessary dimensions and logger.

        Args:
            grid_width (int): The width of the grid.
            grid_height (int): The height of the grid.
            logger (logging.Logger): The logger object for logging messages.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.obstacles: Set[pygame.math.Vector2] = set()

    def calculate_distance(
        self, position1: pygame.math.Vector2, position2: pygame.math.Vector2
    ) -> float:
        """
        Calculate the Euclidean distance between two points in the grid.

        Utilizes Pygame's Vector2 object to calculate the Euclidean distance which is more suitable for
        grid-based pathfinding over Manhattan distance in certain scenarios.

        Args:
            position1 (pygame.math.Vector2): The first position vector.
            position2 (pygame.math.Vector2): The second position vector.

        Returns:
            float: The Euclidean distance between the two positions.
        """
        return position1.distance_to(position2)

    def calculate_obstacle_proximity(
        self, position: pygame.math.Vector2, space_around_obstacles: int = 5
    ) -> float:
        """
        Calculate a penalty score based on the proximity to the nearest obstacle.

        This function iterates through each obstacle and calculates the distance to the given position.
        If the distance is less than the specified space around obstacles, a penalty is calculated based
        on the inverse of the distance to emphasize closer obstacles more significantly.

        Args:
            position (pygame.math.Vector2): The current position as a Vector2 object.
            space_around_obstacles (int): The minimum desired distance from any obstacle.

        Returns:
            float: The total penalty accumulated from all nearby obstacles.
        """
        penalty = 0.0
        for obstacle in self.obstacles:
            distance = self.calculate_distance(position, obstacle)
            if distance <= space_around_obstacles:
                penalty += 1 / (distance + 1)  # Adding 1 to avoid division by zero
        return penalty

    def calculate_boundary_proximity(
        self,
        position: pygame.math.Vector2,
        boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
        space_around_boundaries: int = 5,
    ) -> float:
        """
        Calculate a penalty based on the proximity to boundaries.

        This function computes a penalty score based on how close the given position is to the boundaries of the environment.
        The penalty increases as the position approaches the boundary within a specified threshold.

        Args:
            position (pygame.math.Vector2): The current position as a Vector2 object.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).
            space_around_boundaries (int): The desired space to maintain around boundaries.

        Returns:
            float: The calculated penalty based on proximity to boundaries.
        """
        x_min, y_min, x_max, y_max = boundaries
        # Calculate the minimum distance to any boundary
        min_dist_to_boundary = min(
            position.x - x_min,
            x_max - position.x,
            position.y - y_min,
            y_max - position.y,
        )
        # Apply penalty if within the specified space around boundaries
        if min_dist_to_boundary < space_around_boundaries:
            return (space_around_boundaries - min_dist_to_boundary) ** 2
        return 0.0

    def calculate_body_position_proximity(
        self,
        position: pygame.math.Vector2,
        body_positions: Set[pygame.math.Vector2] = set(),
        space_around_agent: int = 2,
    ) -> float:
        """
        Calculate a penalty for being too close to the snake's own body.

        This function iterates through each position occupied by the snake's body and calculates a penalty
        if the given position is within a specified distance from any part of the body. The penalty is set to infinity
        to represent an impassable barrier.

        Args:
            position (pygame.math.Vector2): The current position as a Vector2 object.
            body_positions (Set[pygame.math.Vector2]): A set of positions occupied by the snake's body as Vector2 objects.
            space_around_agent (int): The desired space to maintain around the snake's body.

        Returns:
            float: The calculated penalty for being too close to the snake's body.
        """
        penalty = 0.0
        for body_pos in body_positions:
            # Calculate distance to each body position
            if self.calculate_distance(position, body_pos) < space_around_agent:
                penalty += float("inf")  # Infinite penalty for collision
        return penalty

    def evaluate_escape_routes(
        self,
        position: pygame.math.Vector2,
        obstacles: Set[pygame.math.Vector2] = set(),
        boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
    ) -> float:
        """
        Evaluate and score the availability of escape routes.

        This function assesses the number of available escape routes from the current position.
        It checks each cardinal direction (up, down, left, right) and scores based on the number of unobstructed paths.

        Args:
            position (pygame.math.Vector2): The current position as a Vector2 object.
            obstacles (Set[pygame.math.Vector2]): A set of obstacle positions as Vector2 objects.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).

        Returns:
            float: The score based on the availability of escape routes.
        """
        score = 0.0
        directions = [
            pygame.math.Vector2(0, 1),
            pygame.math.Vector2(1, 0),
            pygame.math.Vector2(0, -1),
            pygame.math.Vector2(-1, 0),
        ]
        for direction in directions:
            neighbor = position + direction
            # Check if the neighboring position is within boundaries and not an obstacle
            if neighbor not in obstacles and self.is_within_boundaries(
                neighbor, boundaries
            ):
                score += 1.0
        return -score  # Negative score to represent fewer escape routes

    def is_within_boundaries(
        self,
        position: pygame.math.Vector2,
        boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
    ) -> bool:
        """
        Check if a position is within the specified boundaries.

        This function determines whether a given position falls within the defined environmental boundaries.

        Args:
            position (pygame.math.Vector2): The position to check as a Vector2 object.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).

        Returns:
            bool: True if the position is within the boundaries, False otherwise.
        """
        x_min, y_min, x_max, y_max = boundaries
        return x_min <= position.x <= x_max and y_min <= position.y <= y_max

    def apply_zigzagging_effect(self, current_heuristic: float = 1.0) -> float:
        """
        Modify the heuristic to account for zigzagging, making the path less predictable.

        This function increases the heuristic value slightly to account for the added complexity of zigzagging,
        which can make the path less predictable and potentially safer from pursuers.

        Args:
            current_heuristic (float): The current heuristic value.

        Returns:
            float: The modified heuristic value accounting for zigzagging.
        """
        return current_heuristic * 1.05

    def apply_dense_packing_effect(self, current_heuristic: float = 1.0) -> float:
        """
        Modify the heuristic to handle dense packing scenarios more effectively.

        This function decreases the heuristic value to account for dense packing scenarios,
        where closer packing might be necessary or unavoidable.

        Args:
            current_heuristic (float): The current heuristic value.

        Returns:
            float: The modified heuristic value accounting for dense packing.
        """
        return current_heuristic * 0.95

    def heuristic(
        self,
        self_position: Vector2,
        goal_position: Vector2,
        secondary_goal_position: Optional[Vector2] = None,
        tertiary_goal_position: Optional[Vector2] = None,
        quaternary_goal_position: Optional[Vector2] = None,
        environment_boundaries: Tuple[int, int, int, int] = (0, 0, 10, 10),
        space_around_agent: int = 0,
        space_around_goals: int = 0,
        space_around_obstacles: int = 0,
        space_around_boundaries: int = 0,
        obstacles: Set[Vector2] = set(),
        escape_route_availability: bool = False,
        enhancements: List[str] = ["zigzagging"],
        dense_packing: bool = True,
        body_size_adaptations: bool = True,
        self_body_positions: Set[Vector2] = set(),
    ) -> float:
        """
        Calculate the heuristic value for the Dynamic Pathfinding algorithm.

        This heuristic incorporates multiple factors such as directional bias,
        obstacle avoidance, boundary awareness, body avoidance, escape route
        availability, dense packing, and path-specific adjustments. The heuristic
        is designed to generate strategic, efficient paths that adapt to the
        current environment state.

        Args:
            self_position (Vector2): The current position of the agent.
            goal_position (Vector2): The target position the agent aims to reach.
            secondary_goal_position (Optional[Vector2]): Optional secondary target.
            tertiary_goal_position (Optional[Vector2]): Optional tertiary target.
            quaternary_goal_position (Optional[Vector2]): Optional quaternary target.
            environment_boundaries (Tuple[int, int, int, int]): The environment boundaries.
            space_around_agent (int): Space to consider around the agent for planning.
            space_around_goals (int): Space to consider around goals for planning.
            space_around_obstacles (int): Space to consider around obstacles for planning.
            space_around_boundaries (int): Space to consider around boundaries for planning.
            obstacles (Set[Vector2]): The positions of obstacles in the environment.
            escape_route_availability (bool): Whether to consider escape routes.
            enhancements (List[str]): Enhancements to apply to the path.
            dense_packing (bool): Whether to consider dense packing scenarios.
            body_size_adaptations (bool): Whether to consider the agent's body size.
            self_body_positions (Set[Vector2]): Positions occupied by the agent's body.

        Returns:
            float: The calculated heuristic value for the current state.
        """
        # Initialize the heuristic value
        heuristic_value = 0.0

        # Calculate the distance to the primary goal and any secondary goals
        heuristic_value += self.calculate_distance(self_position, goal_position)
        if secondary_goal_position:
            heuristic_value += 0.5 * self.calculate_distance(
                self_position, secondary_goal_position
            )
        if tertiary_goal_position:
            heuristic_value += 0.3 * self.calculate_distance(
                self_position, tertiary_goal_position
            )
        if quaternary_goal_position:
            heuristic_value += 0.1 * self.calculate_distance(
                self_position, quaternary_goal_position
            )

        # Adjust heuristic based on the proximity to obstacles and boundaries
        heuristic_value += self.calculate_obstacle_proximity(
            self_position, obstacles, space_around_obstacles
        )
        heuristic_value += self.calculate_boundary_proximity(
            self_position, environment_boundaries, space_around_boundaries
        )

        # Consider agent's body positions if body size adaptations are enabled
        if body_size_adaptations:
            heuristic_value += self.calculate_body_position_proximity(
                self_position, self_body_positions, space_around_agent
            )

        # Factor in escape routes availability
        if escape_route_availability:
            heuristic_value += self.evaluate_escape_routes(
                self_position, obstacles, environment_boundaries
            )

        # Apply enhancements to the heuristic calculation
        for enhancement in enhancements:
            if enhancement == "zigzagging":
                heuristic_value = self.apply_zigzagging_effect(heuristic_value)
            elif enhancement == "dense_packing":
                heuristic_value = self.apply_dense_packing_effect(heuristic_value)

        # Log the calculated heuristic value
        self.logger.debug(f"Calculated heuristic value: {heuristic_value}")

        return heuristic_value

    def a_star_search(self, start: Vector2, goal: Vector2) -> List[Vector2]:
        """
        Implement the A* algorithm to find the optimal path from start to goal.

        Args:
            start (Vector2): The starting position.
            goal (Vector2): The goal position.

        Returns:
            List[Vector2]: The optimal path from start to goal as a list of positions.
        """
        # Initialize open set and add the starting position with its heuristic value
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))

        # Initialize dictionaries to store the path and cost information
        came_from = {}
        cost_so_far = {start: 0}

        while open_set:
            # Get the position with the lowest cost from the open set
            _, current_cost, current = heapq.heappop(open_set)

            # Check if the goal position is reached
            if current == goal:
                return self.reconstruct_path(came_from, start, goal)

            # Explore the neighbors of the current position
            for next_position in self.neighbors(current):
                # Calculate the new cost to reach the next position
                new_cost = current_cost + self.heuristic(next_position, goal)

                # Update the cost and path if a better path is found
                if (
                    next_position not in cost_so_far
                    or new_cost < cost_so_far[next_position]
                ):
                    cost_so_far[next_position] = new_cost
                    priority = new_cost + self.heuristic(next_position, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_position))
                    came_from[next_position] = current

        # Return an empty path if no path is found
        return []

    def reconstruct_path(
        self,
        came_from: Dict[Vector2, Vector2],
        start: Vector2,
        goal: Vector2,
    ) -> List[Vector2]:
        """
        Reconstruct the path from start to goal using the came_from map.

        Args:
            came_from (Dict[Vector2, Vector2]): Dictionary mapping each position to its previous position.
            start (Vector2): The starting position.
            goal (Vector2): The goal position.

        Returns:
            List[Vector2]: The reconstructed path from start to goal as a list of positions.
        """
        # Initialize an empty path
        path = []

        # Start from the goal position and trace back to the start position
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]

        # Add the start position to the path
        path.append(start)

        # Reverse the path to get the correct order from start to goal
        path.reverse()

        return path

    def neighbors(self, node: Vector2) -> List[Vector2]:
        """
        Generate the neighbors of a node considering boundaries and obstacles.

        Args:
            node (Vector2): The current node position.

        Returns:
            List[Vector2]: A list of neighboring positions.
        """
        # Define the possible directions for neighbors
        directions = [Vector2(0, 1), Vector2(1, 0), Vector2(0, -1), Vector2(-1, 0)]

        # Initialize an empty list to store the neighbors
        result = []

        # Iterate over each direction
        for direction in directions:
            # Calculate the new position by adding the direction to the current node
            new_position = node + direction

            # Check if the new position is within the boundaries and not an obstacle
            if (
                0 <= new_position.x < self.width
                and 0 <= new_position.y < self.height
                and new_position not in self.obstacles
            ):
                result.append(new_position)

        return result


def get_neighbors(
    node: Vector2, width: int, height: int, obstacles: Set[Vector2]
) -> List[Vector2]:
    """
    Generate the neighbors of a node considering boundaries and obstacles.

    Args:
        node (Vector2): The current node position.
        width (int): The width of the grid.
        height (int): The height of the grid.
        obstacles (Set[Vector2]): A set of obstacle positions.

    Returns:
        List[Vector2]: A list of neighboring positions.
    """
    # Define the possible directions for neighbors
    directions = [Vector2(0, 1), Vector2(1, 0), Vector2(0, -1), Vector2(-1, 0)]

    # Initialize an empty list to store the neighbors
    result = []

    # Iterate over each direction
    for direction in directions:
        # Calculate the new position by adding the direction to the current node
        new_position = node + direction

        # Check if the new position is within the boundaries and not an obstacle
        if (
            0 <= new_position.x < width
            and 0 <= new_position.y < height
            and new_position not in obstacles
        ):
            result.append(new_position)

    return result
