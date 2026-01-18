# Rest of the Imports required in alignment with the rest of the classes.
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint

# Initialize Pygame
pg.init()
# Initialize the display
pg.display.init()
# Retrieve the current display information
display_info = pg.display.Info()


# Calculate the block size based on screen resolution to ensure visibility and proportionality
# Define a scaling function for block size relative to screen resolution
def calculate_block_size(screen_width: int, screen_height: int) -> int:
    # Define the reference resolution and corresponding block size
    reference_resolution = (1920, 1080)
    reference_block_size = 20

    # Calculate the scaling factor based on the reference
    scaling_factor_width = screen_width / reference_resolution[0]
    scaling_factor_height = screen_height / reference_resolution[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    # Calculate the block size dynamically based on the screen size
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))

    # Ensure the block size does not become too large or too small
    # Set minimum block size to 1x1 pixels and maximum to 30x30 pixels
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


# Apply the calculated block size based on the current screen resolution
BLOCK_SIZE = calculate_block_size(display_info.current_w, display_info.current_h)

# Define the border width as equivalent to 3 blocks
border_width = 3 * BLOCK_SIZE  # Width of the border to be subtracted from each side

# Define the screen size with a proportional border around the edges
SCREEN_SIZE = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)

# Define a constant for the border color as solid white
BORDER_COLOR = (255, 255, 255)  # RGB color code for white
SNAKE = snake.Snake()
APPLE = apple.Apple()
SEARCH = search.Search(SNAKE, APPLE)
CLOCK = pg.time.Clock()
FPS = 60
TICK_RATE = 1000 // FPS


# // Setting up everything
# Initial setup of the game environment
# function setup() {
def setup() -> (
    Tuple[pg.Surface, snake.Snake, apple.Apple, search.Search, pg.time.Clock]
):
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    # Initialize Pygame
    pg.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pg.Surface = pg.display.set_mode(SCREEN_SIZE)
    # Instantiate the Snake object using the globally defined SNAKE
    snake: snake.Snake = SNAKE
    # Instantiate the Apple object using the globally defined APPLE
    apple: apple.Apple = APPLE
    # Instantiate the Search object with snake and apple as parameters using the globally defined SEARCH
    search: search.Search = SEARCH
    # Initiate the pathfinding algorithm
    search.get_path()
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock: pg.time.Clock = CLOCK
    return screen, snake, apple, search, clock


class Pathfinder:
    def __init__(self, width: int, height: int, logger: logging.Logger):
        """
        Initialize the Pathfinder class with grid dimensions and a logger.

        Args:
            width: The width of the grid.
            height: The height of the grid.
            logger: The logger object for logging messages.
        """
        self.width = width
        self.height = height
        self.logger = logger
        self.obstacles: Set[Tuple[int, int]] = set()

    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate the Manhattan distance between two points.

        Args:
            pos1: The first position as a tuple of (x, y) coordinates.
            pos2: The second position as a tuple of (x, y) coordinates.

        Returns:
            The Manhattan distance between the two points.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def calculate_obstacle_proximity(
        self,
        position: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        space_around_obstacles: int,
    ) -> float:
        """
        Calculate a penalty based on the proximity to obstacles.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            obstacles: A set of obstacle positions as tuples of (x, y) coordinates.
            space_around_obstacles: The desired space to maintain around obstacles.

        Returns:
            The calculated penalty based on proximity to obstacles.
        """
        penalty = 0.0
        for obstacle in obstacles:
            distance = self.calculate_distance(position, obstacle)
            if distance <= space_around_obstacles:
                penalty += 1 / (distance + 1)
        return penalty

    def calculate_boundary_proximity(
        self,
        position: Tuple[int, int],
        boundaries: Tuple[int, int, int, int],
        space_around_boundaries: int,
    ) -> float:
        """
        Calculate a penalty based on the proximity to boundaries.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).
            space_around_boundaries: The desired space to maintain around boundaries.

        Returns:
            The calculated penalty based on proximity to boundaries.
        """
        x_min, y_min, x_max, y_max = boundaries
        min_dist_to_boundary = min(
            position[0] - x_min,
            x_max - position[0],
            position[1] - y_min,
            y_max - position[1],
        )
        if min_dist_to_boundary < space_around_boundaries:
            return (space_around_boundaries - min_dist_to_boundary) ** 2
        return 0.0

    def calculate_body_position_proximity(
        self,
        position: Tuple[int, int],
        body_positions: Set[Tuple[int, int]],
        space_around_agent: int,
    ) -> float:
        """
        Calculate a penalty for being too close to the snake's own body.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            body_positions: A set of positions occupied by the snake's body as tuples of (x, y) coordinates.
            space_around_agent: The desired space to maintain around the snake's body.

        Returns:
            The calculated penalty for being too close to the snake's body.
        """
        penalty = 0.0
        for body_pos in body_positions:
            if self.calculate_distance(position, body_pos) < space_around_agent:
                penalty += float("inf")
        return penalty

    def evaluate_escape_routes(
        self,
        position: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        boundaries: Tuple[int, int, int, int],
    ) -> float:
        """
        Evaluate and score the availability of escape routes.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            obstacles: A set of obstacle positions as tuples of (x, y) coordinates.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).

        Returns:
            The score based on the availability of escape routes.
        """
        score = 0.0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            neighbor = (position[0] + dx, position[1] + dy)
            if neighbor not in obstacles and self.is_within_boundaries(
                neighbor, boundaries
            ):
                score += 1.0
        return -score

    def is_within_boundaries(
        self, position: Tuple[int, int], boundaries: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if a position is within the specified boundaries.

        Args:
            position: The position to check as a tuple of (x, y) coordinates.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).

        Returns:
            True if the position is within the boundaries, False otherwise.
        """
        x_min, y_min, x_max, y_max = boundaries
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max

    def apply_zigzagging_effect(self, current_heuristic: float) -> float:
        """
        Modify the heuristic to account for zigzagging, making the path less predictable.

        Args:
            current_heuristic: The current heuristic value.

        Returns:
            The modified heuristic value accounting for zigzagging.
        """
        return current_heuristic * 1.05

    def apply_dense_packing_effect(self, current_heuristic: float) -> float:
        """
        Modify the heuristic to handle dense packing scenarios more effectively.

        Args:
            current_heuristic: The current heuristic value.

        Returns:
            The modified heuristic value accounting for dense packing.
        """
        return current_heuristic * 0.95

    def heuristic(
        self,
        self_position: Tuple[int, int],
        goal_position: Tuple[int, int],
        secondary_goal_position: Optional[Tuple[int, int]] = None,
        tertiary_goal_position: Optional[Tuple[int, int]] = None,
        quaternary_goal_position: Optional[Tuple[int, int]] = None,
        environment_boundaries: Tuple[int, int, int, int] = (0, 0, 10, 10),
        space_around_agent: int = 0,
        space_around_goals: int = 0,
        space_around_obstacles: int = 0,
        space_around_boundaries: int = 0,
        obstacles: Set[Tuple[int, int]] = set(),
        escape_route_availability: bool = False,
        enhancements: List[str] = ["zigzagging"],
        dense_packing: bool = True,
        body_size_adaptations: bool = True,
        self_body_positions: Set[Tuple[int, int]] = set(),
    ) -> float:
        """
        Calculate the heuristic value for the Dynamic Pathfinding algorithm using a dynamic, adaptive, multifaceted approach.

        This heuristic is optimized for real-time performance and scalability, incorporating multiple factors such as
        directional bias, obstacle avoidance, boundary awareness, body avoidance, escape route availability, dense packing,
        and path-specific adjustments. The heuristic is designed to generate strategic, efficient paths that adapt to the
        current surrounding/grid/environment/game state and adjust accordingly and efficiently.

        Args:
            self_position: The current position of the agent in the grid.
            goal_position: The target position the agent aims to reach.
            secondary_goal_position: An optional secondary target position.
            tertiary_goal_position: An optional tertiary target position.
            quaternary_goal_position: An optional quaternary target position.
            environment_boundaries: The boundaries of the grid/environment/game state.
            space_around_agent: The space around the agent to consider for path planning.
            space_around_goals: The space around the goals to consider for path planning.
            space_around_obstacles: The space around the obstacles to consider for path planning.
            space_around_boundaries: The space around the boundaries to consider for path planning.
            obstacles: The positions of obstacles in the grid.
            escape_route_availability: Whether escape routes should be considered.
            enhancements: The enhancements to apply to the path.
            dense_packing: Whether dense packing scenarios should be considered.
            body_size_adaptations: Whether the agent's body size should be considered.
            self_body_positions: The positions occupied by the agent's body (if any).

        Returns:
            The calculated heuristic value for the current state.
        """
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

    def a_star_search(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Implement the A* algorithm to find the optimal path from start to goal.

        Args:
            start: The starting position as a tuple of (x, y) coordinates.
            goal: The goal position as a tuple of (x, y) coordinates.

        Returns:
            The optimal path from start to goal as a list of (x, y) coordinates.
        """
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, start, goal)

            for next_pos in self.neighbors(current):
                new_cost = current_cost + self.heuristic(next_pos, goal)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_pos))
                    came_from[next_pos] = current

        return []

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal using the came_from map.

        Args:
            came_from: A dictionary mapping each position to its previous position in the path.
            start: The starting position as a tuple of (x, y) coordinates.
            goal: The goal position as a tuple of (x, y) coordinates.

        Returns:
            The reconstructed path from start to goal as a list of (x, y) coordinates.
        """
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate the neighbors of a node considering boundaries and obstacles.

        Args:
            node: The current node as a tuple of (x, y) coordinates.

        Returns:
            A list of neighboring positions as tuples of (x, y) coordinates.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = []
        for dx, dy in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if (
                0 <= nx < self.width
                and 0 <= ny < self.height
                and (nx, ny) not in self.obstacles
            ):
                result.append((nx, ny))
        return result


def get_neighbors(
    node: Tuple[int, int], width: int, height: int, obstacles: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Generate the neighbors of a node considering boundaries and obstacles.

    Args:
        node: The current node as a tuple of (x, y) coordinates.
        width: The width of the grid.
        height: The height of the grid.
        obstacles: A set of obstacle positions as tuples of (x, y) coordinates.

    Returns:
        A list of neighboring positions as tuples of (x, y) coordinates.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
            result.append((nx, ny))
    return result


# Simulation and additional extra stuff. ignore for now.
'''
    def run_simulation():
        """
        Run a simulation of the pathfinding algorithm with dynamic, programmatic simulated grids and agents.

        This function generates random start and goal positions, obstacles, and agent body configurations.
        It then finds the optimal path using the A* algorithm and displays the path using Pygame.
        The simulation runs for a total of 50 iterations, each with a different grid configuration.
        Obstacles are dynamically updated and paths are recalculated in real-time for each iteration.
        """
        pygame.init()
        screen_width, screen_height = 1200, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()

        pathfinder = Pathfinder(
            screen_width // 10, screen_height // 10, logging.getLogger()
        )
        num_scenarios = 50
        display_duration = 5  # Duration in seconds to display each scenario

        for scenario in range(num_scenarios):
            start = (
                random.randint(0, pathfinder.width - 1),
                random.randint(0, pathfinder.height - 1),
            )
            goal = (
                random.randint(0, pathfinder.width - 1),
                random.randint(0, pathfinder.height - 1),
            )

            # Generate complex obstacles
            num_obstacles = random.randint(50, 150)
            obstacles = {
                (
                    random.randint(0, pathfinder.width - 1),
                    random.randint(0, pathfinder.height - 1),
                )
                for _ in range(num_obstacles)
            }

            # Generate agent body positions
            agent_body_size = random.randint(1, 10)
            agent_body_positions = set()
            current_pos = start
            for _ in range(agent_body_size):
                agent_body_positions.add(current_pos)
                current_pos = random.choice(
                    get_neighbors(
                        current_pos, pathfinder.width, pathfinder.height, obstacles
                    )
                )

            # Initialize path
            path = pathfinder.a_star_search(start, goal)

            # Initialize timer
            start_time = pygame.time.get_ticks()

            while pygame.time.get_ticks() - start_time < display_duration * 1000:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Update obstacles dynamically
                obstacles = update_obstacles(obstacles, pathfinder.width, pathfinder.height)

                # Recalculate path
                path = pathfinder.a_star_search(start, goal)

                # Drawing routine
                screen.fill((0, 0, 0))
                for y in range(pathfinder.height):
                    for x in range(pathfinder.width):
                        if (x, y) in obstacles:
                            pygame.draw.rect(
                                screen, (255, 255, 255), (x * 10, y * 10, 10, 10)
                            )
                        if (x, y) in agent_body_positions:
                            pygame.draw.rect(screen, (0, 255, 0), (x * 10, y * 10, 10, 10))
                        if (x, y) in path:
                            pygame.draw.rect(screen, (0, 0, 255), (x * 10, y * 10, 10, 10))
                pygame.draw.rect(
                    screen, (255, 0, 0), (start[0] * 10, start[1] * 10, 10, 10)
                )
                pygame.draw.rect(screen, (0, 255, 0), (goal[0] * 10, goal[1] * 10, 10, 10))

                # Display scenario number and elapsed time
                font = pygame.font.Font(None, 36)
                scenario_text = font.render(
                    f"Scenario: {scenario + 1}/{num_scenarios}", True, (255, 255, 255)
                )
                time_text = font.render(
                    f"Time: {(pygame.time.get_ticks() - start_time) / 1000:.1f}s",
                    True,
                    (255, 255, 255),
                )
                screen.blit(scenario_text, (10, 10))
                screen.blit(time_text, (10, 50))

                pygame.display.flip()
                clock.tick(60)

            # Pause before moving to the next scenario
            pygame.time.delay(1000)



    This is only needed when running the simulation to test the pathfinding explicitly



    def update_obstacles(
        obstacles: Set[Tuple[int, int]], width: int, height: int
    ) -> Set[Tuple[int, int]]:
        """
        Update the obstacles dynamically by adding new random obstacles.

        Args:
            obstacles: A set of obstacle positions as tuples of (x, y) coordinates.
            width: The width of the grid.
            height: The height of the grid.

        Returns:
            An updated set of obstacle positions as tuples of (x, y) coordinates.
        """
        new_obstacles = {
            (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            )
            for _ in range(5)
        }
        obstacles.update(new_obstacles)
        return obstacles
    class Pathfinder:
        """
        This class epitomizes a universal, advanced, dynamic, efficient, and robust pathfinding algorithm, meticulously engineered to be universally applicable across a broad spectrum of pathfinding contexts. It integrates a comprehensive, detailed, and exhaustive set of parameters to ensure optimal pathfinding capabilities. These parameters are meticulously designed to cater to the most complex scenarios, ensuring adaptability and precision. The parameters considered include, but are not limited to:

        - current_node_coordinates: Tuple[int, int] (default: (0, 0))
            The precise current position of the agent within the game grid or environment, represented as a tuple of integers for grid coordinates. The default value is the origin point of the grid, representing a typical starting position.

        - body_occupied_nodes: Set[Tuple[int, int]] (default: set())
            A set of tuples representing grid coordinates occupied by the agent's body, crucial in scenarios where the agent's body can obstruct its path. The default is an empty set, indicating no initial body occupation.

        - goal_node_coordinates: Tuple[int, int] (default: (10, 10))
            The primary target position the agent aims to reach, specified as a tuple of integers for exact grid coordinates. The default target is set to (10, 10), a common goal location in a standard grid.

        - obstacle_positions: Set[Tuple[int, int]] (default: set())
            A set of tuples indicating the positions of static or dynamic obstacles within the environment that must be navigated around. The default is an empty set, indicating a clear grid.

        - environment_boundaries: Tuple[int, int, int, int] (default: (0, 0, 100, 100))
            The boundaries of the playable or navigable area, typically defined by minimum and maximum coordinates (x_min, y_min, x_max, y_max), ensuring precise boundary management. The default represents a 100x100 grid.

        - secondary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
            An optional secondary target position, specified as a tuple of integers for grid coordinates. The default is None, indicating no secondary goal.

        - tertiary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
            An optional tertiary target position, enhancing the complexity and adaptability of the pathfinding. The default is None, indicating no tertiary goal.

        - quaternary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
            An optional quaternary target position, further extending the flexibility and utility of the pathfinding algorithm. The default is None, indicating no quaternary goal.

        - space_around_agent: int (default: 1)
            The required clearance around the agent, specified in grid units, to avoid collisions, ensuring safe navigation. The default clearance is 1 grid unit.

        - space_around_goals: Dict[str, int] (default: {'primary': 1, 'secondary': 1, 'tertiary': 1, 'quaternary': 1})
            A dictionary mapping goal identifiers to specific clearance requirements around primary and optional secondary, tertiary, and quaternary goals, ensuring tailored pathfinding. Each goal has a default clearance of 1 grid unit.

        - space_around_obstacles: int (default: 1)
            The required clearance around obstacles, specified in grid units, to ensure safe navigation and obstacle avoidance. The default clearance is 1 grid unit.

        - space_around_boundaries: int (default: 1)
            The minimum distance, in grid units, the agent must maintain from the boundaries to avoid leaving the navigable area, ensuring boundary compliance. The default distance is 1 grid unit.

        - path_count: int (default: 3)
            The number of alternative paths to generate and evaluate, providing options for optimal path selection. The default is 3 paths, offering a balance between computational efficiency and choice.

        - path_granularity: int (default: 1)
            The level of detail or fineness in the generated paths, affecting the smoothness and precision of navigation, specified in grid units. The default granularity is 1, which corresponds to the finest level of detail.

        - update_frequency: int (default: 100)
            The frequency, in milliseconds, at which the pathfinding algorithm updates the paths based on dynamic changes in the environment, ensuring real-time adaptability. The default frequency is 100 milliseconds.

        - escape_route_availability: bool (default: True)
            A boolean indicating whether the algorithm should calculate escape routes in case of sudden blockages or threats, enhancing safety and robustness. The default is True, enabling escape route calculations.

        - dense_packing: bool (default: False)
            A boolean specifying whether the pathfinding should consider dense packing scenarios, relevant in tightly packed, multi-agent environments, ensuring efficient space utilization. The default is False, assuming a less crowded environment.

        - path_enhancements: List[str] (default: [])
            A list of specific enhancements or modifications to the path logic, such as zigzagging, maintaining clearance, etc., tailored to enhance pathfinding efficiency and adaptability. The default is an empty list, indicating standard pathfinding without enhancements.

        - body_size_adaptations: bool (default: False)
            A boolean indicating adjustments in the pathfinding logic based on the size of the agent's body, affecting how the space around the agent is calculated, ensuring size-specific pathfinding. The default is False, assuming a standard body size.

        - last_direction_moved: Optional[Tuple[int, int]] (default: None)
            The last movement direction of the agent, used to introduce biases or penalties in the path calculation to prevent oscillations or repetitive patterns, enhancing path stability. The default is None, indicating no prior movement bias.

        - environmental_adaptability: bool (default: True)
            A boolean indicating whether the pathfinding logic dynamically adapts to changes in the environmental conditions, such as weather, terrain type, and other dynamic factors, ensuring robust adaptability. The default is True, enabling dynamic adaptation.

        - multi_agent_coordination: bool (default: False)
            A boolean specifying whether the pathfinding algorithm is designed to coordinate paths among multiple agents, crucial in scenarios involving team-based or competitive multi-agent systems. The default is False, tailored for single-agent scenarios.

        - resource_optimization: bool (default: True)
            A boolean indicating whether the pathfinding algorithm optimizes the use of computational and other resources, ensuring efficient operation even under constraints. The default is True, promoting resource-efficient pathfinding.

        This comprehensive parameter set ensures that the Pathfinder class can be adapted and utilized in a wide range of scenarios, promoting modular, flexible, clean, scalable development and maintenance. Each parameter is intricately designed to interact synergistically, providing a robust framework that supports dynamic, efficient, and precise pathfinding capabilities.

        The Pathfinder class employs a variety of sophisticated algorithms, including but not limited to A* for optimal pathfinding, Dijkstra's algorithm for shortest path calculations, and custom heuristic functions tailored for specific scenarios such as densely packed environments or dynamic obstacle navigation. These algorithms are integrated with advanced data structures for state management and efficient computation, such as priority queues for open set management in A*, and hash tables for quick look-up of node states.

        The class is engineered to handle a multitude of environmental variables and agent characteristics. It dynamically adjusts its computations based on the agent's body size, the proximity of obstacles, and the designated goals. This adaptability is achieved through a meticulously crafted set of methods that assess and respond to real-time changes in the environment, such as sudden appearance of obstacles or changes in the agent's body configuration.

        Furthermore, the Pathfinder class is designed with multi-agent coordination in mind, allowing for the seamless integration of pathfinding strategies among multiple agents in a shared environment. This is particularly crucial in competitive or cooperative scenarios where agents must navigate without colliding with each other while still achieving their individual or collective goals.

        Resource optimization is a key aspect of the Pathfinder's design, ensuring that the pathfinding operations are not only accurate but also computationally efficient. This allows the Pathfinder to be deployed in systems with limited computational resources while still maintaining high performance and responsiveness.

        In summary, the Pathfinder class is a paragon of modern pathfinding techniques, encapsulating a wide array of functionalities that are both comprehensive and customizable. It stands as a testament to advanced, flexible, and efficient software design, capable of adapting to and excelling in any pathfinding scenario presented.
        """


    class Pathfinder:
        """
        This class epitomizes a universal, advanced, dynamic, efficient, and robust pathfinding algorithm, meticulously engineered to be universally applicable across a broad spectrum of pathfinding contexts. It integrates a comprehensive, detailed, and exhaustive set of parameters to ensure optimal pathfinding capabilities. These parameters are meticulously designed to cater to the most complex scenarios, ensuring adaptability and precision. The parameters considered include, but are not limited to:

        - current_node_coordinates: Tuple[int, int] (default: (0, 0))
            The precise current position of the agent within the game grid or environment, represented as a tuple of integers for grid coordinates. The default value is the origin point of the grid, representing a typical starting position.

        - body_occupied_nodes: Set[Tuple[int, int]] (default: set())
            A set of tuples representing grid coordinates occupied by the agent's body, crucial in scenarios where the agent's body can obstruct its path. The default is an empty set, indicating no initial body occupation.

        - goal_node_coordinates: Tuple[int, int] (default: (10, 10))
            The primary target position the agent aims to reach, specified as a tuple of integers for exact grid coordinates. The default target is set to (10, 10), a common goal location in a standard grid.

        - obstacle_positions: Set[Tuple[int, int]] (default: set())
            A set of tuples indicating the positions of static or dynamic obstacles within the environment that must be navigated around. The default is an empty set, indicating a clear grid.

        - environment_boundaries: Tuple[int, int, int, int] (default: (0, 0, 100, 100))
            The boundaries of the playable or navigable area, typically defined by minimum and maximum coordinates (x_min, y_min, x_max, y_max), ensuring precise boundary management. The default represents a 100x100 grid.

        - secondary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
            An optional secondary target position, specified as a tuple of integers for grid coordinates. The default is None, indicating no secondary goal.

        - tertiary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
            An optional tertiary target position, enhancing the complexity and adaptability of the pathfinding. The default is None, indicating no tertiary goal.

        - quaternary_goal_coordinates: Optional[Tuple[int, int]] (default: None)
            An optional quaternary target position, further extending the flexibility and utility of the pathfinding algorithm. The default is None, indicating no quaternary goal.

        - space_around_agent: int (default: 1)
            The required clearance around the agent, specified in grid units, to avoid collisions, ensuring safe navigation. The default clearance is 1 grid unit.

        - space_around_goals: Dict[str, int] (default: {'primary': 1, 'secondary': 1, 'tertiary': 1, 'quaternary': 1})
            A dictionary mapping goal identifiers to specific clearance requirements around primary and optional secondary, tertiary, and quaternary goals, ensuring tailored pathfinding. Each goal has a default clearance of 1 grid unit.

        - space_around_obstacles: int (default: 1)
            The required clearance around obstacles, specified in grid units, to ensure safe navigation and obstacle avoidance. The default clearance is 1 grid unit.

        - space_around_boundaries: int (default: 1)
            The minimum distance, in grid units, the agent must maintain from the boundaries to avoid leaving the navigable area, ensuring boundary compliance. The default distance is 1 grid unit.

        - path_count: int (default: 3)
            The number of alternative paths to generate and evaluate, providing options for optimal path selection. The default is 3 paths, offering a balance between computational efficiency and choice.

        - path_granularity: int (default: 1)
            The level of detail or fineness in the generated paths, affecting the smoothness and precision of navigation, specified in grid units. The default granularity is 1, which corresponds to the finest level of detail.

        - update_frequency: int (default: 100)
            The frequency, in milliseconds, at which the pathfinding algorithm updates the paths based on dynamic changes in the environment, ensuring real-time adaptability. The default frequency is 100 milliseconds.

        - escape_route_availability: bool (default: True)
            A boolean indicating whether the algorithm should calculate escape routes in case of sudden blockages or threats, enhancing safety and robustness. The default is True, enabling escape route calculations.

        - dense_packing: bool (default: False)
            A boolean specifying whether the pathfinding should consider dense packing scenarios, relevant in tightly packed, multi-agent environments, ensuring efficient space utilization. The default is False, assuming a less crowded environment.

        - path_enhancements: List[str] (default: [])
            A list of specific enhancements or modifications to the path logic, such as zigzagging, maintaining clearance, etc., tailored to enhance pathfinding efficiency and adaptability. The default is an empty list, indicating standard pathfinding without enhancements.

        - body_size_adaptations: bool (default: False)
            A boolean indicating adjustments in the pathfinding logic based on the size of the agent's body, affecting how the space around the agent is calculated, ensuring size-specific pathfinding. The default is False, assuming a standard body size.

        - last_direction_moved: Optional[Tuple[int, int]] (default: None)
            The last movement direction of the agent, used to introduce biases or penalties in the path calculation to prevent oscillations or repetitive patterns, enhancing path stability. The default is None, indicating no prior movement bias.

        - environmental_adaptability: bool (default: True)
            A boolean indicating whether the pathfinding logic dynamically adapts to changes in the environmental conditions, such as weather, terrain type, and other dynamic factors, ensuring robust adaptability. The default is True, enabling dynamic adaptation.

        - multi_agent_coordination: bool (default: False)
            A boolean specifying whether the pathfinding algorithm is designed to coordinate paths among multiple agents, crucial in scenarios involving team-based or competitive multi-agent systems. The default is False, tailored for single-agent scenarios.

        - resource_optimization: bool (default: True)
            A boolean indicating whether the pathfinding algorithm optimizes the use of computational and other resources, ensuring efficient operation even under constraints. The default is True, promoting resource-efficient pathfinding.

        This comprehensive parameter set ensures that the Pathfinder class can be adapted and utilized in a wide range of scenarios, promoting modular, flexible, clean, scalable development and maintenance. Each parameter is intricately designed to interact synergistically, providing a robust framework that supports dynamic, efficient, and precise pathfinding capabilities.

        The Pathfinder class employs a variety of sophisticated algorithms, including but not limited to A* for optimal pathfinding, Dijkstra's algorithm for shortest path calculations, and custom heuristic functions tailored for specific scenarios such as densely packed environments or dynamic obstacle navigation. These algorithms are integrated with advanced data structures for state management and efficient computation, such as priority queues for open set management in A*, and hash tables for quick look-up of node states.

        The class is engineered to handle a multitude of environmental variables and agent characteristics. It dynamically adjusts its computations based on the agent's body size, the proximity of obstacles, and the designated goals. This adaptability is achieved through a meticulously crafted set of methods that assess and respond to real-time changes in the environment, such as sudden appearance of obstacles or changes in the agent's body configuration.

        Furthermore, the Pathfinder class is designed with multi-agent coordination in mind, allowing for the seamless integration of pathfinding strategies among multiple agents in a shared environment. This is particularly crucial in competitive or cooperative scenarios where agents must navigate without colliding with each other while still achieving their individual or collective goals.

        Resource optimization is a key aspect of the Pathfinder's design, ensuring that the pathfinding operations are not only accurate but also computationally efficient. This allows the Pathfinder to be deployed in systems with limited computational resources while still maintaining high performance and responsiveness.

        In summary, the Pathfinder class is a paragon of modern pathfinding techniques, encapsulating a wide array of functionalities that are both comprehensive and customizable. It stands as a testament to advanced, flexible, and efficient software design, capable of adapting to and excelling in any pathfinding scenario presented.
        """


    
    from typing import Set, Tuple, Dict, List, Optional
    import heapq
    import logging


    class Pathfinder:
        def __init__(
            self,
            current_node_coordinates: Tuple[int, int] = (0, 0),
            body_occupied_nodes: Set[Tuple[int, int]] = set(),
            goal_node_coordinates: Tuple[int, int] = (10, 10),
            obstacle_positions: Set[Tuple[int, int]] = set(),
            environment_boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
            secondary_goal_coordinates: Optional[Tuple[int, int]] = None,
            tertiary_goal_coordinates: Optional[Tuple[int, int]] = None,
            quaternary_goal_coordinates: Optional[Tuple[int, int]] = None,
            space_around_agent: int = 1,
            space_around_goals: Dict[str, int] = {
                "primary": 1,
                "secondary": 1,
                "tertiary": 1,
                "quaternary": 1,
            },
            space_around_obstacles: int = 1,
            space_around_boundaries: int = 1,
            path_count: int = 3,
            path_granularity: int = 1,
            update_frequency: int = 100,
            escape_route_availability: bool = True,
            dense_packing: bool = False,
            path_enhancements: List[str] = [],
            body_size_adaptations: bool = False,
            last_direction_moved: Optional[Tuple[int, int]] = None,
            environmental_adaptability: bool = True,
            multi_agent_coordination: bool = False,
            resource_optimization: bool = True,
        ):

            # Initial parameter settings
            self.current_node = current_node_coordinates
            self.body_occupied = body_occupied_nodes
            self.goal = goal_node_coordinates
            self.obstacles = obstacle_positions
            self.bounds = environment_boundaries
            self.secondary_goal = secondary_goal_coordinates
            self.tertiary_goal = tertiary_goal_coordinates
            self.quaternary_goal = quaternary_goal_coordinates
            self.space_agent = space_around_agent
            self.space_goals = space_around_goals
            self.space_obstacles = space_around_obstacles
            self.space_boundaries = space_around_boundaries
            self.path_count = path_count
            self.granularity = path_granularity
            self.update_freq = update_frequency
            self.escape_routes = escape_route_availability
            self.dense_packing = dense_packing
            self.enhancements = path_enhancements
            self.body_size_adapt = body_size_adaptations
            self.last_move = last_direction_moved

            # Logger setup
            self.logger = logging.getLogger("Pathfinder")
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        def a_star_search(
            self, start: Tuple[int, int], goal: Tuple[int, int]
        ) -> List[Tuple[int, int]]:
            """Perform A* algorithm to find the optimal path from start to goal."""
            open_set = []
            heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
            came_from = {}
            cost_so_far = {start: 0}

            while open_set:
                _, current_cost, current = heapq.heappop(open_set)

                if current == goal:
                    return self.reconstruct_path(came_from, start, goal)

                for next in self.neighbors(current):
                    new_cost = current_cost + self.cost(current, next)
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + self.heuristic(next, goal)
                        heapq.heappush(open_set, (priority, new_cost, next))
                        came_from[next] = current
            return []

        def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
            """Calculate Manhattan distance as a heuristic for pathfinding."""
            (x1, y1), (x2, y2) = a, b
            return abs(x1 - x2) + abs(y1 - y2)

        def cost(self, from_node: Tuple[int, int], to_node: Tuple[int, int]) -> int:
            """Return the cost of moving between two nodes, considering obstacles."""
            if to_node in self.obstacles:
                return float("inf")  # Impassable
            return 1  # Uniform cost for simplicity

        def neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
            """Generate the neighbors of a node considering boundaries and body occupation."""
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            result = []
            for dx, dy in directions:
                nx, ny = node[0] + dx, node[1] + dy
                if (
                    0 <= nx <= self.bounds[2]
                    and 0 <= ny <= self.bounds[3]
                    and (nx, ny) not in self.body_occupied
                ):
                    result.append((nx, ny))
            return result

        def reconstruct_path(self, came_from, start, goal):
            """Reconstruct the path from start to goal using the came_from map."""
            path = []
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)  # optional
            path.reverse()  # optional
            return path

        def update_obstacles(self, new_obstacles: Set[Tuple[int, int]]):
            """Dynamically update the obstacle positions on the grid."""
            self.obstacles = new_obstacles
            self.logger.debug(f"Obstacles updated: {self.obstacles}")

        def update_agent_body(self, new_body_positions: Set[Tuple[int, int]]):
            """Update the positions occupied by the agent's body."""
            self.body_occupied = new_body_positions
            self.logger.debug(f"Agent body updated: {self.body_occupied}")

        def multi_agent_path_request(self, other_agents: Dict[int, Tuple[int, int]]):
            """Consider paths of other agents to avoid collisions in multi-agent environments."""
            self.logger.debug("Evaluating multi-agent path interactions.")
            # Implementation could involve calculating safe intervals or communication protocols.

        def optimize_path(self, path: List[Tuple[int, int]]):
            """Apply smoothing and optimization techniques to the generated path."""
            self.logger.debug(f"Path before optimization: {path}")
            optimized_path = self.smooth_path(path)
            self.logger.debug(f"Path after optimization: {optimized_path}")
            return optimized_path

        def smooth_path(self, path: List[Tuple[int, int]]):
            """Smoothen the path to minimize sharp turns and potentially reduce path length."""
            if not path:
                return path
            smoothed_path = [path[0]]
            for i in range(1, len(path) - 1):
                if not (
                    path[i - 1][0] == path[i + 1][0] or path[i - 1][1] == path[i + 1][1]
                ):
                    smoothed_path.append(path[i])
            smoothed_path.append(path[-1])
            return smoothed_path
        '''
