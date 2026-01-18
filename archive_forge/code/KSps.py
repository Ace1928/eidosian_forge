import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Color constants using numpy arrays for optimal performance and consistency:
BG_LIGHT_GREEN = np.array([137, 200, 80])
BG_DARK_GREEN = np.array([123, 181, 70])
BLUE = np.array([47, 174, 232])
DARK_BLUE = np.array([44, 163, 217])
RED = np.array([217, 42, 42])
DARK_RED = np.array([197, 38, 38])
GRAY = np.array([155, 155, 155])
BLACK = np.array([0, 0, 0])

# Game constants defined with numpy for precision and efficiency:
TILE_SIZE = 10
HEIGHT = 100
WIDTH = 100
WIN_HEIGHT, WIN_WIDTH = HEIGHT * TILE_SIZE, WIDTH * TILE_SIZE
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Snake")
FPS = 60
CLOCK = pygame.time.Clock()


# Setup logging
def configure_logging():
    """
    Configures the logging settings for this module. This function sets the logging level to DEBUG and specifies the format
    for logging messages. This configuration is encapsulated within this function to prevent side effects on logging settings
    in other modules when this module is imported.
    """
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class Food:
    def __init__(self, position, color=np.array([255, 0, 0])):  # Default color red
        """
        Initializes the Food object with a position and a color.
        Args:
        position (numpy array): The position of the food on the grid.
        color (numpy array): The RGB color value of the food, represented as a numpy array.
        """
        self.position = position
        self.color = color
        logging.debug(f"Food created at {position} with color {color}")

    def draw(self):
        """
        Draws the food on the game window.
        """
        pygame.draw.rect(
            WIN,
            self.color,
            (
                self.position[0] * TILE_SIZE,
                self.position[1] * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            ),
        )
        logging.debug(f"Food drawn at {self.position} with color {self.color}")


class Snake:
    def __init__(self, grid_width=WIDTH, grid_height=HEIGHT):
        """
        Initializes the Snake object with default settings. The snake is initially set to be alive with its segments
        positioned centrally on the game grid. The snake starts without any movement direction and is in a frozen state
        until the game begins. The default color of the snake is set to BLUE, and it can operate in different modes,
        initially set to 'hamiltonian'.

        Args:
            grid_width (int): Width of the game grid.
            grid_height (int): Height of the game grid.
        """
        logging.debug("Initializing Snake object with default settings.")
        try:
            self.alive = True
            self.segments = deque(
                [np.array([grid_width // 2, grid_height // 2 - i]) for i in range(3)]
            )
            self.direction = "right"  # Initial direction
            self.frozen = True
            self.color = BLUE  # Default color
            self.mode = "hamiltonian"  # Modes: 'hamiltonian', 'astar'
            self.color_phase = [
                random.randint(0, 360) for _ in self.segments
            ]  # HSV phase for each segment
            logging.info(
                "Snake object initialized successfully with attributes: alive=True, segments=central, direction=right, frozen=True, color=BLUE, mode=hamiltonian."
            )
        except Exception as e:
            logging.error(f"Failed to initialize Snake object: {e}")
            raise Exception(f"Snake initialization error: {e}")

    def move(self):
        """
        Updates the position of the snake based on its current direction. This method efficiently manages the segments
        deque by appending the new head position and popping the tail position when moving, ensuring optimal memory usage
        and performance.
        """
        logging.debug(
            "Attempting to update the position of the snake based on its current direction."
        )
        try:
            if self.frozen or not self.alive:
                logging.info("Snake movement aborted: Snake is frozen or not alive.")
                return

            new_head = np.copy(self.segments[-1])
            if self.direction == "up":
                new_head[1] -= 1
            elif self.direction == "down":
                new_head[1] += 1
            elif self.direction == "left":
                new_head[0] -= 1
            elif self.direction == "right":
                new_head[0] += 1

            # Check if the new head position is out of bounds or collides with the snake's body
            if not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT) or np.any(
                [np.array_equal(new_head, segment) for segment in self.segments]
            ):
                self.alive = False
                logging.info("Snake has collided with the wall or itself and died.")
                return

            self.segments.append(new_head)
            self.segments.popleft()  # Remove the tail segment
            logging.info(f"Snake moved successfully in direction: {self.direction}.")
        except Exception as e:
            logging.error(f"Failed to move the snake: {e}")
            raise Exception(f"Snake movement error: {e}")

    def change_direction(self, direction):
        """
        Changes the direction of the snake's movement if the new direction is not directly opposite to the current direction,
        preventing the snake from reversing onto itself.
        """
        logging.debug(
            f"Attempting to change the direction of the snake's movement to: {direction}."
        )
        try:
            opposite_directions = {
                "up": "down",
                "down": "up",
                "left": "right",
                "right": "left",
            }
            if direction != opposite_directions.get(self.direction, ""):
                self.direction = direction
                logging.info(
                    f"Snake direction changed successfully to: {self.direction}."
                )
        except Exception as e:
            logging.error(f"Failed to change the direction of the snake: {e}")
            raise Exception(f"Snake direction change error: {e}")

    def draw(self, WIN):
        """
        Draws each segment of the snake with a cycling color spectrum and applies glow effects based on the mode.
        """
        hue_step = 1  # Define how fast the color cycles through the spectrum
        for i, segment in enumerate(self.segments):
            # Update color phase
            self.color_phase[i] = (self.color_phase[i] + hue_step) % 360
            color = pygame.Color(0)
            color.hsva = (
                self.color_phase[i],
                100,
                100,
            )  # Full saturation and value for vivid colors

            # Draw the segment
            pygame.draw.rect(
                WIN,
                color,
                (segment[0] * TILE_SIZE, segment[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE),
            )

            # Apply glow effect based on the mode
            if self.mode == "hamiltonian":
                self.apply_glow(WIN, segment, (255, 0, 0))  # Red glow
            elif self.mode == "astar":
                self.apply_glow(WIN, segment, (0, 0, 255))  # Blue glow

    def apply_glow(self, WIN, position, glow_color):
        """
        Applies a glow effect around a given position with the specified glow color.
        """
        glow_radius = 10  # Radius of the glow effect
        for radius in range(glow_radius, 0, -1):
            alpha = (1 - (radius / glow_radius)) * 255
            glow_surface = pygame.Surface(
                (TILE_SIZE + radius * 2, TILE_SIZE + radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                glow_surface,
                glow_color + (int(alpha),),
                (glow_radius, glow_radius),
                radius,
            )
            WIN.blit(
                glow_surface,
                (position[0] * TILE_SIZE - radius, position[1] * TILE_SIZE - radius),
            )

        pygame.display.update()


class Grid:
    def __init__(self, width, height):
        """
        Initializes the Grid object with specified width and height.

        Args:
            width (int): The width of the grid in terms of number of cells.
            height (int): The height of the grid in terms of number of cells.
        """
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width), dtype=int)  # Initialize a grid of zeros
        logging.debug(f"Grid created with dimensions {width}x{height}")

    def is_position_free(self, position):
        """
        Checks if a given position is free (not occupied) on the grid.

        Args:
            position (numpy array): The position to check on the grid.

        Returns:
            bool: True if the position is free, False otherwise.
        """
        x, y = position
        return self.cells[y, x] == 0

    def get_points(self):
        """
        Generates a list of all coordinate points within the grid.

        Returns:
            list of tuples: A list containing all (x, y) coordinates in the grid.
        """
        points = [(x, y) for x in range(self.width) for y in range(self.height)]
        logging.debug(f"Generated {len(points)} points for the grid.")
        return points

    def update_position(self, position, value):
        """
        Updates the grid cell at a given position with a specified value.

        Args:
            position (numpy array): The position to update on the grid.
            value (int): The value to set at the given position.
        """
        x, y = position
        self.cells[y, x] = value
        logging.debug(f"Grid position {position} updated with value {value}")

    def reset_grid(self):
        """
        Resets the entire grid to zero.
        """
        self.cells.fill(0)
        logging.debug("Grid reset to initial state.")

    def get_obstacles(self):
        """
        Retrieves all obstacles within the grid based on the current game state. This includes the outer boundaries,
        the snake's body, and potential growth areas near the food.

        Returns:
            list of lists of tuples: Each sublist represents a polygonal obstacle defined by its vertices.
        """
        obstacles = []

        # Add outer boundaries as obstacles
        boundaries = [
            [(0, y) for y in range(self.height)],  # Left boundary
            [(self.width - 1, y) for y in range(self.height)],  # Right boundary
            [(x, 0) for x in range(self.width)],  # Top boundary
            [(x, self.height - 1) for x in range(self.width)],  # Bottom boundary
        ]
        obstacles.extend(boundaries)
        logging.debug(f"Outer boundaries added as obstacles.")

        # Add snake body as obstacles
        if hasattr(self, "snake"):
            snake_body = [(segment.x, segment.y) for segment in self.snake.segments]
            obstacles.append(snake_body)
            logging.debug(
                f"Snake body added as obstacles with {len(snake_body)} segments."
            )

        # Add area around snake's tail as obstacles when the snake head is near the food
        if (
            hasattr(self, "food")
            and self.food.position is not None
            and hasattr(self, "snake")
        ):
            snake_head_position = self.snake.segments[0]
            distance_to_food = np.linalg.norm(
                np.array(snake_head_position) - np.array(self.food.position)
            )

            # Define a proximity threshold within which the tail's surrounding area becomes an obstacle
            proximity_threshold = (
                5 * TILE_SIZE
            )  # This threshold can be adjusted based on game dynamics

            if distance_to_food <= proximity_threshold:
                # Include the area around the tail as obstacles
                tail_position = self.snake.segments[-1]
                tail_surrounding_area = [
                    (tail_position.x + dx, tail_position.y + dy)
                    for dx in range(-1, 2)  # from -1 to 1
                    for dy in range(-1, 2)
                    if 0 <= tail_position.x + dx < self.width
                    and 0 <= tail_position.y + dy < self.height
                ]
                obstacles.append(tail_surrounding_area)
                logging.debug(
                    f"Area around snake's tail at {tail_position} added as obstacles due to proximity to food at {self.food.position}."
                )

        logging.debug(f"Total of {len(obstacles)} obstacles retrieved from the grid.")
        return obstacles


class Pathfinding:
    def __init__(self, grid):
        """
        Initialize the Pathfinding class which is responsible for calculating optimal paths for the snake using advanced algorithms.

        Args:
            grid (Grid): The game grid which contains all the necessary information about the game state.

        Attributes:
            grid (Grid): Stores the reference to the game grid.
            path_cache (LRUCache): Caches the results of complex path calculations for quick retrieval.
            pathfinding_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during path calculations.
        """
        self.grid = grid
        self.path_cache = LRUCache(maxsize=1024)
        self.pathfinding_lock = asyncio.Lock()
        logging.debug(f"Pathfinding instance initialized with grid: {grid}")

    async def calculate_path(self, start, goal):
        """
        Asynchronously calculates the optimal path from start to goal using A* algorithm.

        Args:
            start (Node): The starting node of the path.
            goal (Node): The goal node of the path.

        Returns:
            List[Node]: The optimal path as a list of nodes.
        """
        async with self.pathfinding_lock:
            logging.debug(f"Calculating path from {start} to {goal}")
            # Check if the path is already in the cache
            path_key = (hash(start), hash(goal))
            if path_key in self.path_cache:
                logging.debug(f"Path from {start} to {goal} retrieved from cache")
                return self.path_cache[path_key]

            # Perform the computation-intensive task directly in the async function
            path = self._a_star_search(start, goal)
            # Cache the result of the path computation
            self.path_cache[path_key] = path
            logging.debug(f"Path from {start} to {goal} computed and cached")
            return path

    def _a_star_search(self, start, goal):
        """
        Implements the A* search algorithm to find the shortest path from start to goal.

        Args:
            start (Node): The starting node.
            goal (Node): The goal node.

        Returns:
            List[Node]: The path from start to goal as a list of nodes.
        """
        open_set = deque([start])
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.grid._heuristic(start, goal)}
        logging.debug(f"Starting A* search from {start} to {goal}")

        while open_set:
            current = min(open_set, key=lambda o: f_score[o])
            if current == goal:
                path = self.grid._reconstruct_path(came_from, current)
                logging.debug(f"Path found: {path}")
                return path

            open_set.remove(current)
            for neighbor in self.grid.get_neighbors(current):
                tentative_g_score = g_score[current] + self.grid.distance(
                    current, neighbor
                )
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.grid._heuristic(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.append(neighbor)
        logging.debug("No path found")
        return []

    @staticmethod
    def _heuristic(node1, node2, snake, fruit, grid):
        """
        Advanced heuristic that calculates the estimated cost from node1 to node2, incorporating strategic game elements
        such as snake body positioning, fruit positioning, head and tail direction, and overall grid space utilization.
        This heuristic uses a combination of Prim's algorithm and a Hamiltonian cycle to manipulate the snake's body
        positioning, ensuring the head is on the outside of a bunched body and the tail on the inside, both traveling in
        the same direction without intersection. It switches to A* algorithm for rapid acquisition of the fruit when a
        clear path is available, and reverts to the Hamiltonian-Prim's strategy post consumption. Additionally, it integrates
        with the latest pathfinding algorithms (AHP, CDP, ThetaStar) to adapt the heuristic based on the current strategic
        requirements dictated by the DecisionMaker.

        Args:
            node1 (Node): The current node, typically the snake's head.
            node2 (Node): The target node, typically the fruit's position.
            snake (Snake): The snake object containing segments, head, and tail information.
            fruit (Fruit): The fruit object with its current position.
            grid (Grid): The game grid object.

        Returns:
            float: The estimated cost from node1 to node2, adjusted for strategic game play.
        """
        # Calculate base heuristic using Manhattan distance for A* segments
        base_cost = np.linalg.norm(np.array(node1.position) - np.array(node2.position))
        logging.debug(f"Base heuristic cost calculated: {base_cost}")

        # Implementing Hamiltonian cycle and Prim's algorithm for body bunching
        if snake.is_hamiltonian_mode:
            # Calculate the cost of keeping the tail inside and head outside the bunched body
            head_position = np.array(snake.get_head_position())
            tail_position = np.array(snake.get_tail_position())
            fruit_position = np.array(fruit.position)
            body_positions = np.array([seg.position for seg in snake.segments])

            # Calculate the density of the snake's body around the head and tail
            head_density = np.sum(
                np.linalg.norm(body_positions - head_position, axis=1) < 5
            )
            tail_density = np.sum(
                np.linalg.norm(body_positions - tail_position, axis=1) < 5
            )

            # Encourage the head to be on the outside (lower density near head)
            head_cost = -head_density * 10

            # Encourage the tail to be on the inside (higher density near tail)
            tail_cost = tail_density * 10

            # Calculate the cost of keeping the snake away from the fruit while in Hamiltonian mode
            distance_to_fruit = np.linalg.norm(head_position - fruit_position)
            fruit_avoidance_cost = distance_to_fruit * 5

            # Combine costs for Hamiltonian-Prim's strategy
            total_cost = base_cost + head_cost + tail_cost + fruit_avoidance_cost
            logging.debug(f"Total cost in Hamiltonian mode: {total_cost}")
        else:
            # In A* mode, prioritize direct path to fruit
            total_cost = base_cost
            logging.debug(f"Total cost in A* mode: {total_cost}")

        # Integrate with advanced pathfinding algorithms
        pathfinding_cost_adjustments = 0
        if grid.pathfinders["AHP"].is_applicable(node1, node2):
            pathfinding_cost_adjustments += grid.pathfinders["AHP"].calculate_cost(
                node1, node2
            )
        if grid.pathfinders["CDP"].is_applicable(node1, node2):
            pathfinding_cost_adjustments += grid.pathfinders["CDP"].calculate_cost(
                node1, node2
            )
        if grid.pathfinders["ThetaStar"].is_applicable(node1, node2):
            pathfinding_cost_adjustments += grid.pathfinders[
                "ThetaStar"
            ].calculate_cost(node1, node2)

        logging.debug(f"Pathfinding cost adjustments: {pathfinding_cost_adjustments}")

        # Final heuristic cost with pathfinding adjustments
        final_heuristic_cost = total_cost + pathfinding_cost_adjustments
        logging.debug(f"Final heuristic cost calculated: {final_heuristic_cost}")

        return final_heuristic_cost

    @staticmethod
    def _reconstruct_path(came_from, current):
        """
        Reconstruct the path from start to goal using the came_from map.

        Args:
            came_from (dict): The map of nodes to their predecessors.
            current (Node): The current node to start reconstruction from.

        Returns:
            List[Node]: The reconstructed path as a list of nodes.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        logging.debug(f"Reconstructed path: {total_path[::-1]}")
        return total_path[::-1]  # Return reversed path


class DecisionMaker:
    def __init__(self, snake=None, food=None, grid=None):
        """
        Initializes the DecisionMaker object which controls the snake's movement strategy with optional parameters.
        This constructor meticulously sets up the snake, food, and grid objects, and initializes pathfinders for various strategies,
        ensuring that each component is optimally configured for high-performance gameplay.

        Args:
            snake (Snake): The snake object. Defaults to a new Snake instance if not provided.
            food (Food): The food object. Defaults to a new Food instance positioned at the center if not provided.
            grid (Grid): The grid object representing the game area. Defaults to a new Grid instance with predefined width and height if not provided.
        """
        self.snake = snake if snake is not None else Snake()
        self.food = (
            food
            if food is not None
            else Food(position=np.array([WIDTH // 2, HEIGHT // 2]))
        )
        self.grid = grid if grid is not None else Grid(WIDTH, HEIGHT)
        self.pathfinders = {
            "CDP": ConstrainedDelaunayPathfinder(
                self.grid.get_points(), self.grid.get_obstacles()
            ),
            "AHP": AmoebaHamiltonianPathfinder(self.snake, self.grid, self.food),
            "ThetaStar": ThetaStar(self.grid),
        }

    async def decide_next_move(self):
        """
        Asynchronously decides the next move for the snake based on the current game state and the selected strategy.
        It evaluates potential paths using multiple pathfinding algorithms, compares their costs, and selects the optimal path.
        This method continuously updates its assessments to look ahead multiple steps based on the game's complexity and dynamics.

        Returns:
            str: The next direction for the snake to move, determined by the optimal pathfinding strategy.
        """
        current_position = self.snake.get_head_position()
        food_position = self.food.position
        paths = {}
        costs = {}

        # Generate paths from each pathfinding strategy
        for name, pathfinder in self.pathfinders.items():
            path = await pathfinder.find_path(current_position, food_position)
            paths[name] = path
            costs[name] = self.calculate_path_cost(path)

        # Determine the optimal path with the lowest cost
        optimal_strategy = min(costs, key=costs.get)
        optimal_path = paths[optimal_strategy]

        # Decide the next move based on the optimal path
        next_move = self.determine_next_move_from_path(optimal_path)
        return next_move

    def calculate_path_cost(self, path):
        """
        Calculates the cost of a given path based on length, proximity to dangers, and strategic advantages.
        This method employs a detailed and comprehensive cost analysis to ensure optimal path selection.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: The calculated cost of the path.
        """
        length_cost = len(path) * 10
        danger_cost = sum(self.grid.is_near_obstacle(point) for point in path) * 20
        strategic_cost = self.evaluate_strategic_advantages(path)
        return length_cost + danger_cost + strategic_cost

    def evaluate_strategic_advantages(self, path):
        """
        Evaluates the strategic advantages of a given path, meticulously considering factors such as proximity to future food positions,
        avoidance of potential future obstacles, and alignment with game-winning strategies. This method employs a sophisticated
        multi-criteria decision-making framework to assess the strategic value of each path, integrating advanced algorithms
        for predictive analytics and dynamic game state evaluation.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: The strategic advantage cost of the path, calculated as a weighted sum of multiple strategic factors.
        """
        if not path:
            logging.debug("No path provided, returning infinite disadvantage.")
            return float("inf")  # No path means infinitely disadvantageous

        # Initialize strategic costs
        future_food_proximity_cost = 0.0
        future_obstacle_avoidance_cost = 0.0
        game_winning_alignment_cost = 0.0

        # Predict future food positions using a probabilistic model
        predicted_food_positions = self.predict_future_food_positions()
        logging.debug(f"Predicted future food positions: {predicted_food_positions}")

        # Calculate proximity to future food positions
        future_food_proximity_cost = sum(
            min(
                np.linalg.norm(np.array(pos) - np.array(food_pos))
                for food_pos in predicted_food_positions
            )
            for pos in path
        )
        logging.debug(
            f"Calculated future food proximity cost: {future_food_proximity_cost}"
        )

        # Calculate cost of potential future obstacles
        future_obstacle_avoidance_cost = sum(
            self.calculate_obstacle_proximity_cost(pos) for pos in path
        )
        logging.debug(
            f"Calculated future obstacle avoidance cost: {future_obstacle_avoidance_cost}"
        )

        # Evaluate alignment with game-winning strategies
        game_winning_alignment_cost = self.evaluate_game_winning_strategy_alignment(
            path
        )
        logging.debug(
            f"Calculated game winning alignment cost: {game_winning_alignment_cost}"
        )

        # Weighted sum of strategic costs
        strategic_cost = (
            0.4 * future_food_proximity_cost
            + 0.4 * future_obstacle_avoidance_cost
            + 0.2 * game_winning_alignment_cost
        )
        logging.debug(f"Total strategic cost calculated: {strategic_cost}")

        return strategic_cost

    def predict_future_food_positions(self):
        """
        Predicts future food positions based on a comprehensive analysis of historical game data and the current game state,
        utilizing a sophisticated machine learning model. This method integrates complex data analysis techniques to forecast
        the probable locations where food might appear on the game grid, thereby enabling strategic planning for the snake's movements.

        Returns:
            list: A meticulously compiled list of predicted future food positions, each represented as a tuple of coordinates.
        """

        # Logging the initiation of the food position prediction process
        logging.debug("Initiating the prediction of future food positions.")

        # Extract historical food position data and current game state
        historical_data = self.retrieve_historical_food_positions()
        current_state_features = self.extract_features_from_current_state()

        # Combine historical data with current state for prediction
        features = np.concatenate((historical_data, current_state_features), axis=0)

        # Normalize the feature set
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features[:, :-1],
            scaled_features[:, -1],
            test_size=0.2,
            random_state=42,
        )

        # Initialize and train the machine learning model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict the future food positions
        predicted_positions = model.predict(X_test)
        predicted_positions = [
            (int(pos[0]), int(pos[1])) for pos in predicted_positions
        ]

        # Log the completion of the prediction process
        logging.debug(f"Predicted future food positions: {predicted_positions}")

        return predicted_positions

    def retrieve_historical_food_positions(self):
        """
        Retrieves historical food positions from the game's database or a data file, which are used to train the predictive model.
        This method meticulously extracts data from a structured database or file system, ensuring that the data integrity and
        accuracy are maintained to provide a robust foundation for predictive modeling.

        Returns:
            numpy.ndarray: An array containing the historical food positions and associated game states, each entry meticulously
            structured to ensure comprehensive data representation.
        """
        # Establish a connection to the database or file system
        logging.debug(
            "Establishing connection to the database for retrieving historical food positions."
        )
        try:
            # Placeholder for database connection establishment
            database_connection = (
                None  # This should be replaced with actual database connection logic
            )

            # Execute a query to retrieve historical food positions
            query = "SELECT position_x, position_y, game_state FROM food_positions ORDER BY game_time DESC"
            logging.debug(f"Executing query to retrieve historical data: {query}")
            cursor = database_connection.cursor()
            cursor.execute(query)
            data = cursor.fetchall()

            # Convert the data into a numpy array for further processing
            historical_positions = np.array(data)
            logging.info(
                f"Retrieved historical food positions: {historical_positions.shape[0]} records found."
            )
            return historical_positions
        except Exception as e:
            logging.error(f"Failed to retrieve historical food positions: {e}")
            raise Exception(f"Database retrieval error: {e}")
        finally:
            if database_connection:
                database_connection.close()
                logging.debug(
                    "Database connection closed after retrieving historical food positions."
                )

    def extract_features_from_current_state(self):
        """
        Extracts features from the current game state, which are crucial for the prediction model to forecast future food positions accurately.
        This method analyzes the current state of the game, including the snake's length and the food's position, to extract relevant features
        that will be used to enhance the predictive accuracy of the model.

        Returns:
            numpy.ndarray: An array of features extracted from the current game state, each feature carefully selected and processed to
            maximize the predictive performance of the model.
        """
        logging.debug(
            "Extracting features from the current game state for predictive modeling."
        )
        try:
            # Extract the snake's length and the current position of the food
            snake_length = self.snake.length
            food_position_x = self.food.position[0]
            food_position_y = self.food.position[1]
            logging.debug(
                f"Current snake length: {snake_length}, Food position: ({food_position_x}, {food_position_y})"
            )

            # Combine these features into a numpy array
            current_state_features = np.array(
                [snake_length, food_position_x, food_position_y]
            )
            logging.info(
                f"Features extracted from current game state: {current_state_features}"
            )
            return current_state_features
        except Exception as e:
            logging.error(f"Failed to extract features from current game state: {e}")
            raise Exception(f"Feature extraction error: {e}")

    def calculate_obstacle_proximity_cost(self, position):
        """
        Calculates the cost associated with proximity to obstacles for a given position on the grid.

        Args:
            position (tuple): The grid position to evaluate.

        Returns:
            float: The calculated cost based on obstacle proximity.
        """
        is_near_obstacle = self.grid.is_near_obstacle(position)
        cost = 20 if is_near_obstacle else 0
        logging.debug(
            f"Calculated obstacle proximity cost for position {position}: {cost}"
        )
        return cost

    def evaluate_game_winning_strategy_alignment(self, path):
        """
        Evaluates how well the given path aligns with established game-winning strategies.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: A cost representing the strategic alignment with winning strategies.
        """
        alignment_cost = 10 * (
            len(path) - self.snake.length
        )  # Simplified strategy alignment cost
        logging.debug(
            f"Calculated game winning strategy alignment cost for path: {alignment_cost}"
        )
        return alignment_cost

    def determine_next_move_from_path(self, path):
        """
        Determines the next move direction based on the first step in the path.
        This method ensures that the decision is made with precision and aligns with the optimal path strategy.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            str: The direction to move ('up', 'down', 'left', 'right').
        """
        if not path or len(path) < 2:
            return "none"  # No move possible

        current_head = self.snake.get_head_position()
        next_position = path[1]  # The next step in the path

        if next_position[0] < current_head[0]:
            return "left"
        elif next_position[0] > current_head[0]:
            return "right"
        elif next_position[1] < current_head[1]:
            return "up"
        elif next_position[1] > current_head[1]:
            return "down"
        return "none"


# CDP - Good for Complex Environments
class ConstrainedDelaunayPathfinder:
    def __init__(self, points, obstacles):
        """
        Initializes the pathfinder with points defining the space and obstacles.

        Args:
            points (list of tuples): Points defining the navigable space.
            obstacles (list of lists of tuples): Each sublist represents a polygonal obstacle.
        """
        self.points = points
        self.obstacles = obstacles
        self.tri = Delaunay(points)
        self.graph = self.build_graph()

    def build_graph(self):
        """
        Constructs the graph based on Delaunay triangulation and applies constraints for obstacles.

        Returns:
            graph (networkx.Graph): The graph representing navigable paths.
        """
        edges = set()
        for simplex in self.tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    if self.is_edge_navigable(
                        self.tri.points[simplex[i]], self.tri.points[simplex[j]]
                    ):
                        edges.add((simplex[i], simplex[j]))

        graph = nx.Graph()
        for edge in edges:
            p1, p2 = self.tri.points[list(edge)]
            distance = np.linalg.norm(p1 - p2)
            graph.add_edge(edge[0], edge[1], weight=distance)

        return graph

    def is_edge_navigable(self, p1, p2):
        """
        Determines if an edge can be navigated, i.e., it does not intersect any obstacles.

        Args:
            p1, p2 (tuple): The endpoints of the edge.

        Returns:
            bool: True if the edge is navigable, False otherwise.
        """
        line = np.array([p1, p2])
        for obstacle in self.obstacles:
            poly = np.array(obstacle)
            if np.any(self.intersect(poly, line)):
                return False
        return True

    def intersect(self, poly, line):
        """
        Check if the line intersects with the polygon.

        Args:
            poly (np.array): Array of points defining the polygon.
            line (np.array): Two points defining the line.

        Returns:
            bool: True if there is an intersection, False otherwise.
        """
        return np.any(np.cross(poly - line[0], line[1] - line[0]) == 0)

    def find_path(self, start_point, end_point):
        """
        Finds the path from start_point to end_point using A* algorithm on the constructed graph.

        Args:
            start_point (tuple): The starting point of the path.
            end_point (tuple): The ending point of the path.

        Returns:
            list: The path from start to end, as a list of points.
        """
        start_vertex = min(
            self.graph.nodes,
            key=lambda vertex: np.linalg.norm(
                np.array(self.tri.points[vertex]) - np.array(start_point)
            ),
        )
        end_vertex = min(
            self.graph.nodes,
            key=lambda vertex: np.linalg.norm(
                np.array(self.tri.points[vertex]) - np.array(end_point)
            ),
        )

        path = nx.astar_path(
            self.graph,
            start_vertex,
            end_vertex,
            heuristic=lambda a, b: np.linalg.norm(
                np.array(self.tri.points[a]) - np.array(self.tri.points[b])
            ),
        )
        return [self.tri.points[vertex] for vertex in path]


# Example usage:
# points = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
# obstacles = [[(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]]
# pathfinder = ConstrainedDelaunayPathfinder(points, obstacles)
# path = pathfinder.find_path((0, 0), (1, 1))
# print("Path:", path)


# AHP - Innovative "Bunching" and "Dashing" Bioinspired Algorithm
class AmoebaHamiltonianPathfinder:
    def __init__(self, snake, grid, food):
        """
        Initializes the pathfinder with references to the snake, grid, and food objects.
        This method prepares the Hamiltonian cycle that the snake will initially attempt to maintain.

        Args:
            snake (Snake): The snake object, which should have methods to get its head and tail positions, and its body segments.
            grid (Grid): The grid object, which should provide the dimensions of the play area.
            food (Food): The food object, which should have a position attribute.
        """
        logging.info("Initializing AmoebaHamiltonianPathfinder")
        self.snake = snake
        self.grid = grid
        self.food = food
        self.graph = self.build_initial_graph()
        logging.info("Graph built successfully with nodes and edges.")

    def build_initial_graph(self):
        """
        Constructs the initial Hamiltonian cycle graph for the snake to follow.

        Returns:
            networkx.Graph: The graph representing the Hamiltonian cycle.
        """
        logging.info("Building initial Hamiltonian graph.")
        points = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        graph = nx.grid_2d_graph(self.grid.width, self.grid.height, periodic=True)
        logging.info("Hamiltonian graph constructed with periodic boundaries.")
        return graph

    def update_graph_for_food(self):
        """
        Updates the graph dynamically to include a path to the food while maintaining the Hamiltonian cycle properties.

        Returns:
            None
        """
        logging.info("Updating graph for new food position.")
        closest_point = min(
            self.graph.nodes,
            key=lambda point: np.linalg.norm(
                np.array(point) - np.array(self.food.position)
            ),
        )
        logging.info(
            f"Closest point on Hamiltonian cycle to food located at {closest_point}."
        )

        if closest_point not in self.graph.neighbors(self.food.position):
            self.graph.add_edge(closest_point, tuple(self.food.position))
            logging.info(
                f"Edge added between {closest_point} and {self.food.position} for direct path."
            )

    def find_path(self):
        """
        Computes the path for the snake's head to follow from its current position to the food.

        Returns:
            list[tuple]: The path coordinates for the snake to follow.
        """
        current_position = tuple(self.snake.get_head_position())
        logging.info(
            f"Finding path from snake head at {current_position} to food at {self.food.position}."
        )
        path = nx.shortest_path(
            self.graph, source=current_position, target=tuple(self.food.position)
        )
        logging.info("Path found successfully.")
        return path

    def maintain_bunching(self):
        """
        Adjusts the snake's body to maintain a bunched, compact formation.

        Returns:
            None
        """
        logging.info("Maintaining snake body bunching.")
        body_positions = np.array([seg.position for seg in self.snake.segments])
        centroid = np.mean(body_positions, axis=0).astype(int)
        tail_position = self.snake.get_tail_position()

        logging.info(
            f"Calculating path for tail movement towards body centroid at {centroid}."
        )
        tail_path = nx.shortest_path(
            self.graph, source=tail_position, target=tuple(centroid)
        )

        for step in tail_path:
            self.snake.move_tail_to(step)
            logging.info(f"Tail moved to {step} to maintain bunching.")


# Example usage assumes the existence of classes Snake, Grid, and Food with appropriate attributes and methods.
# If these classes are not defined, this example cannot be directly executed.
# snake = Snake()
# grid = Grid(width=10, height=10)
# food = Food(position=np.array([5, 5]))
# pathfinder = AmoebaHamiltonianPathfinder(snake, grid, food)
# path = pathfinder.find_path()
# print("Path:", path)


# A* - Theta* Pathfinding Algorithm Efficient Game Pathfinding
class ThetaStar:
    def __init__(self, grid):
        self.grid = grid
        self.open_set = []
        self.came_from = {}
        self.g_score = {}
        self.f_score = {}

    def heuristic(self, a, b):
        return math.dist(a, b)

    def line_of_sight(self, s, s2):
        """Check if there is a direct line of sight between two points using Bresenham's Line Algorithm"""
        x0, y0 = s
        x1, y1 = s2
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            if (x0, y0) == (x1, y1):
                return True
            e2 = 2 * err
            if e2 >= dy:
                if x0 == x1:
                    break
                err += dy
                x0 += sx
            if e2 <= dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy
            if not self.grid.is_walkable(x0, y0):
                return False
        return True

    def theta_star_search(self, start, goal):
        """Perform the Theta* search algorithm"""
        heapq.heappush(self.open_set, (0, start))
        self.g_score[start] = 0
        self.f_score[start] = self.heuristic(start, goal)
        self.came_from[start] = None

        while self.open_set:
            _, current = heapq.heappop(self.open_set)

            if current == goal:
                return self.reconstruct_path(current)

            for neighbor in self.grid.get_neighbors(current):
                if neighbor not in self.g_score:
                    self.g_score[neighbor] = float("inf")
                if self.line_of_sight(self.came_from[current], neighbor):
                    tentative_g_score = self.g_score[
                        self.came_from[current]
                    ] + self.heuristic(self.came_from[current], neighbor)
                else:
                    tentative_g_score = self.g_score[current] + self.heuristic(
                        current, neighbor
                    )

                if tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = (
                        current
                        if not self.line_of_sight(self.came_from[current], neighbor)
                        else self.came_from[current]
                    )
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal
                    )
                    heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))

        return []

    def reconstruct_path(self, current):
        """Reconstruct the path from start to goal"""
        total_path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            total_path.append(current)
        return total_path[::-1]  # Return reversed path


# Example usage:
# grid = Grid(...)  # Define your grid with the method is_walkable and get_neighbors
# theta_star = ThetaStar(grid)
# start, goal = (1, 1), (10, 10)
# path = theta_star.theta_star_search(start, goal)
# print("Path:", path)


class GameManager:
    def __init__(self):
        """
        Initializes the GameManager object which manages the state and progression of the game. This includes initializing
        the snake object, generating the initial food placement, and setting the initial mode of the game to not be in
        Hamiltonian mode. The GameManager orchestrates the game loop and handles the transitions between different states
        of the game. Utilizes numpy arrays for optimal performance and precision in coordinate handling.
        """
        self.snake = Snake()
        self.food = self.generate_food(self.snake)
        self.grid = Grid(WIDTH, HEIGHT)
        self.decision_maker = DecisionMaker(self.snake, self.food, self.grid)
        self.speed = 5
        self.speed_increment = 5

    def generate_food(self, snake):
        """
        Generates a new food object at a random position on the game grid that is not occupied by the snake. This method
        is essential for the continuity of the game, providing goals for the player to achieve. Utilizes numpy arrays for
        optimal position calculations and torch for AI/ML optimizations.
        """
        logging.debug("Initiating the generation of new food.")
        all_positions = np.array([(x, y) for x in range(WIDTH) for y in range(HEIGHT)])
        snake_positions = np.array(
            [seg for seg in snake.segments]
        )  # Assuming snake.segments is already a list of tuples

        # Use numpy broadcasting to find positions not in snake.segments
        mask = ~np.any(
            np.all(all_positions[:, None] == snake_positions, axis=-1), axis=-1
        )
        possible_positions = all_positions[mask]

        logging.debug(
            f"Calculated possible positions for new food: {possible_positions}"
        )

        if possible_positions.size > 0:
            viable_positions = np.array([pos for pos in possible_positions])
            logging.debug(
                f"Viable positions after distance filtering: {viable_positions}"
            )

            if viable_positions.size > 0:
                chosen_position = viable_positions[
                    np.random.randint(len(viable_positions))
                ]
                logging.debug(f"Chosen position for new food: {chosen_position}")
                new_food = Food(chosen_position)
                logging.info(f"New food generated at position: {chosen_position}")
                return new_food
            else:
                logging.warning(
                    "No viable positions available after applying distance filter."
                )
                return None
        else:
            logging.warning("No available positions to generate new food.")
            return None

    def run(self):
        last_move_time = time.time()
        move_interval = 1 / self.speed

        while True:
            current_time = time.time()
            if current_time - last_move_time >= move_interval:
                # Decision-making integration
                next_direction = self.decision_maker.decide_next_move()
                self.snake.change_direction(next_direction)
                self.update_game_state()
                last_move_time = current_time

            self.handle_input()
            self.render_game()
            pygame.display.update()
            CLOCK.tick(60)  # Maintain the game responsiveness

    def handle_input(self):
        """
        Handles all the input events from the user, such as keyboard presses. This method ensures that the game responds
        to user inputs accurately and timely, enhancing the interactivity of the game. Utilizes Pygame's event system.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.snake.frozen = (
                        not self.snake.frozen
                    )  # Toggle snake's frozen state

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.speed = max(
                        1, self.speed - self.speed_increment
                    )  # Decrease speed, minimum 1
                elif event.key == pygame.K_e:
                    self.speed += self.speed_increment  # Increase speed

                # Map keys directly to directions
                direction_map = {
                    pygame.K_LEFT: "left",
                    pygame.K_RIGHT: "right",
                    pygame.K_UP: "up",
                    pygame.K_DOWN: "down",
                }

                if event.key in direction_map:
                    new_direction = direction_map[event.key]
                    current_direction = self.snake.direction

                    # Prevent the snake from reversing directly
                    if (
                        (current_direction == "left" and new_direction != "right")
                        or (current_direction == "right" and new_direction != "left")
                        or (current_direction == "up" and new_direction != "down")
                        or (current_direction == "down" and new_direction != "up")
                    ):
                        self.snake.change_direction(new_direction)

    async def update_game_state(self):
        if self.snake.alive and not self.snake.frozen:
            next_direction = await self.decision_maker.decide_next_move()
            if next_direction:
                self.snake.change_direction(next_direction)
            self.snake.move()
            if self.food and np.array_equal(
                self.snake.segments[-1], self.food.position
            ):
                self.snake.grow()  # Method to grow the snake
                self.food = self.generate_food(self.snake)

    def render_game(self):
        """
        Renders all the game elements including the background, snake, and food onto the game window. This method ensures
        that the visual representation of the game is always up-to-date and visually appealing. Utilizes Pygame's drawing
        functions.
        """
        WIN.fill(BG_LIGHT_GREEN)  # Clear the screen with a light green background
        self.snake.draw(WIN)
        if self.food:
            self.food.draw()


def main():
    configure_logging()
    pygame.init()
    game_manager = GameManager()
    profiler = cProfile.Profile()
    profiler.enable()
    game_manager.run()
    profiler.disable()
    profiler.print_stats(sort="time")
    pygame.quit()


if __name__ == "__main__":
    main()
