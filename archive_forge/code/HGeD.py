"""
Node Class
Methods:
__init__(self, position: Tuple[int, int], cost: float = 0.0, heuristic: float = 0.0)
__lt__(self, other: 'Node')
__eq__(self, other: 'Node')
__hash__(self)
__repr__(self)
__str__(self)
__getitem__(self, key: str)
__setitem__(self, key: str, value: Any)
__iter__(self)
__len__(self)
__contains__(self, item: Any)

Grid Class
Methods:
__init__(self, width: int, height: int, tile_size: int)
generate_fruit(self)
get_neighbors(self, node: Node)
can_extend(self, node: Node)
get_fruit_location(self)
get_grid(self)
get_width(self)
get_height(self)
get_tile_size(self)
get_tile_color(self, x: int, y: int)
get_tile_value(self, x: int, y: int)

Tile Class
Methods:
__init__(self, x: int, y: int, size: int, color: Tuple[int, int, int])
draw(self, surface: pygame.Surface)

Fruit Class
Methods:
__init__(self, position: Tuple[int, int], size: int, color: Tuple[int, int, int], border_color: Tuple[int, int, int], border_thickness: int)
draw(self, surface: pygame.Surface)
update(self)
get_position(self)
get_size(self)
get_color(self)
get_border_color(self)
get_border_thickness(self)
get_tile_size(self)
get_tile_color(self)
get_tile_value(self)
get_fruit_location(self)
get_neighbors(self)
can_extend(self)
get_width(self)
get_height(self)


Snake Class
Methods:
__init__(self, segments: Deque[Node], color: Tuple[int, int, int])
draw(self, surface: pygame.Surface)
move(self, direction: Tuple[int, int])
change_direction(self, new_direction: Tuple[int, int])
get_direction(self)
get_segments(self)
get_color(self)
set_color(self, new_color: Tuple[int, int, int])
get_mode(self)
set_mode(self, mode: str)
get_frozen(self)
set_frozen(self, frozen: bool)
get_alive(self)
set_alive(self, alive: bool)
get_length(self)
get_head_position(self)
get_tail_position(self)
get_body_positions(self)
is_position_in_body(self, position: Tuple[int, int])
is_position_out_of_bounds(self, position: Tuple[int, int])

Pathfinding Class
Methods:
__init__(self)
calculate_hamiltonian_cycle(self, grid: Grid)
a_star_search(self, start: Node, goal: Node, grid: Grid)
follow_hamiltonian(self)
switch_to_a_star(self)
update_snake_color(self, snake: Snake)
get_direction_array(self)
is_ordered(self, nodes: List[Node])
move_snake(self, snake: Snake)
reconstruct_path(self, start: Node, goal: Node) -> List[Node]: Reconstructs the path from start to goal.
is_head_on_outside_of_coil(self, snake: Snake)
is_path_from_head_to_fruit_clear(self, snake: Snake, fruit: Fruit)
get_fruit_location(self, fruit: Fruit)
astar_hamiltonian_heuristic(self, node: Node, goal: Node)
astar_or_hamiltonian_path(self, snake: Snake, fruit: Fruit)

Perception Class
Methods:
__init__(self, grid: Grid)
evaluate_paths(self)
update_perception(self, snake: Snake, fruit: Fruit)
get_possible_moves(self, snake: Snake)
predict_collision(self, path: List[Node])

Decision Class
Methods:
__init__(self)
integrate_data(self, pathfinding: Pathfinding, perception: Perception)
make_decision(self)
train_decision_model(self, data: np.ndarray)
evaluate_decision_accuracy(self)

GUI Class
Methods:
__init__(self)
setup_elements(self)
update_elements(self)
draw_elements(self, surface: pygame.Surface)

Renderer Class
Methods:
__init__(self)
apply_strobe_effects(self, surface: pygame.Surface)
apply_rainbow_effects(self, surface: pygame.Surface)
update_display(self, surface: pygame.Surface)
render_scene(self, game_objects: List[Any], surface: pygame.Surface)

Game Logic Class
Methods:
__init__(self)
handle_events(self, events: List[pygame.event.Event])
update_game_state(self)
check_collisions(self)
check_for_death(self)
spawn_fruit(self)
control_game_speed(self, speed: int)
manage_inputs(self, input_data: Dict[str, Any])
freeze_game(self)
unfreeze_game(self)
pause_game(self)
unpause_game(self)
change_snake_color(self, color: Tuple[int, int, int])

Updater Class
Methods:
__init__(self)
update_all(self)
handle_errors(self)
log_activity(self, message: str)
ensure_graceful_exit(self)

1. Initialization
Game Setup:
The Game Logic Class initializes the game environment using its __init__ method.
The GUI Class sets up the graphical user interface elements through setup_elements.
The Renderer Class is initialized to handle graphical rendering.
Game Entities Initialization:
The Grid Class is instantiated, setting up the game grid with specified dimensions and tile size.
The Snake Class is created with initial segments and color.
The Fruit Class is initialized at a random position on the grid.
2. Game Loop Execution
Event Handling:
The Game Logic Class continuously checks for user inputs and other game events in handle_events.
Game State Updates:
The Updater Class calls update_all to manage updates across all components.
The Snake Class updates its position based on user input or AI decisions using move and change_direction.
The Fruit Class may update its state or position if needed.
AI Computation:
The Decision Class integrates data from Pathfinding and Perception to make decisions on the snake's movements.
The Pathfinding Class calculates paths using algorithms like A or Hamiltonian cycles.
The Perception Class evaluates potential paths and predicts collisions.
3. Rendering and Effects
Visual Updates:
The Renderer Class applies visual effects such as apply_strobe_effects and apply_rainbow_effects.
The GUI Class updates GUI elements as necessary with update_elements.
The Renderer Class then renders the scene including all game objects using render_scene.
4. Collision and Game State Checks
Collision Detection:
The Game Logic Class checks for collisions between the snake and walls, or the snake and itself in check_collisions.
Game Progression:
Check if the snake has eaten the fruit using coordinates comparison, and if so, spawn a new fruit using spawn_fruit and extend the snake.
Game Over Conditions:
The Game Logic Class checks for conditions that would end the game, such as snake collisions, in check_for_death.
5. Game Speed and Control Management
Speed Adjustments:
The Game Logic Class may adjust the game speed based on certain conditions or player inputs using control_game_speed.
Game Pausing/Unpausing:
Handle pausing and unpausing the game through pause_game and unpause_game.
6. Logging and Error Handling
Activity Logging:
The Updater Class logs significant game activities or errors using log_activity.
Error Management:
The Updater Class also handles any errors that occur during the game's execution in handle_errors.
7. Game Termination
Graceful Exit:
Ensure a graceful exit from the game using ensure_graceful_exit in the Updater Class, which would handle any cleanup or final logging needed.
This step-by-step logic flow outlines how each component and method in the advanced Snake game architecture contributes to the game's operation, from initialization through gameplay to termination, ensuring a cohesive and comprehensive game experience.

Recommended Order for Class Implementation:
1. Node Class
2. Grid Class
3. Tile Class
4. Fruit Class
5. Snake Class
6. Pathfinding Class
7. Perception Class
Decision Class
9. Renderer Class
10. GUI Class
11. Game Logic Class
12. Updater Class
Recommended Order for Method Implementation within Each Class:
Initialization methods 1
Utility methods (like getters and setters)
Core functionality methods (like move, draw, update)
AI and decision-making methods
Rendering and GUI update methods
Event handling and game state management methods
Logging and error handling methods
"""

"""
Advanced Snake Game AI and Pathfinding System Module Overview
Module Description:
This module implements an advanced AI-driven Snake game leveraging state-of-the-art technologies and methodologies in machine learning, pathfinding, and multi-agent systems. The system is designed to optimize decision-making processes, enhance gameplay dynamics, and provide a robust framework for continuous learning and adaptation.
Technologies and Libraries Used:
Pygame: For rendering game graphics and handling user interactions.
NumPy: Utilized for efficient numerical computations, especially in AI calculations and data manipulations.
Deque from Collections: Used for efficient FIFO operations on the snake's body segments.
Multiprocessing and Threading: Employed to handle complex computations like pathfinding and AI decision-making concurrently, improving performance.
Asynchronous Processing: Applied in game state updates and rendering to maintain smooth gameplay.
Caching: Used to store pre-computed paths and AI decisions to speed up repeated calculations.
Arrays Instead of Loops: Wherever possible, loops are replaced with vectorized operations using NumPy to enhance performance.
PyTorch: For implementing and training neural networks involved in AI decision-making and continuous learning.
Detailed Implementation:
Pathfinding:
A and Hamiltonian Cycle:
The system alternates between A and Hamiltonian cycle algorithms based on the game state. A is used for short-term goal-oriented pathfinding, while Hamiltonian cycles provide a complete tour of the grid ensuring no repeated nodes until the cycle is broken.
Heuristic for Alternation:
The decision to switch between A and Hamiltonian is based on a heuristic that evaluates the current length of the snake, the proximity to the fruit, and the density of the snake's body on the grid.
If the snake's length is less than a threshold relative to the grid size, A is favored to aggressively pursue the fruit.
As the snake grows, a Hamiltonian path is computed to avoid self-collision and maximize area coverage.
Transition points are determined by analyzing potential path overlaps and the risk of trapping the snake's head.
AI Decision Making:
Neural Network:
A convolutional neural network (CNN) is implemented using PyTorch, designed to evaluate the game grid as input and output decision probabilities for each possible move.
The network architecture consists of several convolutional layers to capture spatial hierarchies, followed by fully connected layers that output move probabilities.
Training:
The network is trained on historical game data, including successful and unsuccessful games. Each training instance consists of the game grid state, the decision made, and the outcome.
Continuous learning is implemented by retraining the network periodically on new game logs, allowing the AI to adapt to new strategies and improve over time.
Integration with Pathfinding:
The neural network's output is integrated with the pathfinding system. The network influences the decision by applying a weighted bias to the paths generated by A or Hamiltonian algorithms.
This hybrid approach allows the AI to choose paths that not only follow the optimal route according to the pathfinding algorithm but also consider strategic positioning against potential future states.
Multi-Agent Coordination (Future Scope):
While the current system focuses on a single-agent scenario (the snake), the architecture is designed to be scalable to multi-agent environments.
Potential extensions could involve competitive or cooperative multi-snake scenarios, where each snake is controlled by an independent AI agent trained to interact within the shared environment.
Continuous Improvement and Monitoring:
Log Analysis:
All moves and decisions made by the AI are logged extensively, including game state, decision rationale, and outcomes.
These logs are analyzed to identify patterns, successes, and areas for improvement.
Feedback Loop:
Insights from log analysis are fed back into the training process for the neural network, ensuring that the AI continuously evolves and adapts to new challenges and strategies.
This module represents a comprehensive and detailed implementation of an advanced AI system for the Snake game, leveraging cutting-edge technologies and methodologies to create a dynamic, intelligent, and continuously improving gameplay experience.
"""
# Imports
import pygame
import numpy as np
from collections import deque
from typing import List, Tuple, Deque, Dict, Any, Optional
import multiprocessing as mp
import threading
import time
import random
import math
import asyncio
import os
import logging
import sys
import aiofiles
from functools import lru_cache as LRUCache
import aiohttp
import json
import cachetools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.utils.data.distributed as distributed
import torch.distributions as distributions
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.cuda as cuda  # Added for potential GPU acceleration
import torch.backends.cudnn as cudnn  # Added for optimizing deep learning computations on CUDA
import logging  # For detailed logging of operations and errors
import hashlib  # For generating unique identifiers for nodes
import bisect  # For maintaining sorted lists
import gc  # For explicit garbage collection if necessary

# Ensure that all redundant imports are removed and necessary imports for performance and optimization are included.


class GlobalPool:
    _instance = None

    @classmethod
    def get_pool(cls):
        if cls._instance is None:
            cls._instance = mp.Pool(processes=mp.cpu_count())
        return cls._instance

    @classmethod
    def close_pool(cls):
        if cls._instance is not None:
            cls._instance.close()
            cls._instance.join()
            cls._instance = None


# Global Pool
GLOBAL_POOL = GlobalPool.get_pool()


# Node Class
class Node:
    def __init__(
        self, position: Tuple[int, int], cost: float = 0.0, heuristic: float = 0.0
    ):
        """
        Initialize a Node instance with position, cost, and heuristic values.

        Args:
            position (Tuple[int, int]): The (x, y) coordinates of the node on the grid.
            cost (float): The cost of reaching this node from the start node.
            heuristic (float): The estimated cost from this node to the goal node.

        Attributes:
            position (Tuple[int, int]): Stores the coordinates of the node.
            cost (float): Stores the cost of reaching this node.
            heuristic (float): Stores the heuristic estimate to the goal.
            neighbors (List[Node]): List of neighboring nodes.
            id (str): Unique identifier for the node.
        """
        self.position = np.array(
            position, dtype=np.int32
        )  # Convert position to a NumPy array for efficient computation
        self.cost = np.float32(
            cost
        )  # Ensure the cost is stored as a float32 for precision and memory efficiency
        self.heuristic = np.float32(
            heuristic
        )  # Ensure the heuristic is stored as a float32 for precision and memory efficiency
        self.neighbors = cachetools.LRUCache(
            maxsize=8
        )  # Initialize a Least Recently Used Cache for neighbors to optimize memory usage
        self.id = hashlib.sha256(
            str(position).encode()
        ).hexdigest()  # Generate a unique identifier using SHA-256 hashing of the position

    def __lt__(self, other: "Node") -> bool:
        """
        Less than comparison for priority queue operations based on cost and heuristic.

        Args:
            other (Node): Another node to compare against.

        Returns:
            bool: True if this node's f-score (cost + heuristic) is less than the other's.
        """
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other: "Node") -> bool:
        """
        Equality check based on node's unique identifier.

        Args:
            other (Node): Another node to compare against.

        Returns:
            bool: True if both nodes have the same identifier.
        """
        return self.id == other.id

    def __hash__(self) -> int:
        """
        Hash function for node based on its unique identifier.

        Returns:
            int: Hash value of the node.
        """
        return hash(self.id)

    def __repr__(self) -> str:
        """
        Official string representation of the Node.

        Returns:
            str: String representation showing position, cost, and heuristic.
        """
        return f"Node(position={self.position.tolist()}, cost={self.cost}, heuristic={self.heuristic})"

    def __str__(self) -> str:
        """
        Informal string representation of the Node.

        Returns:
            str: Simple string showing position.
        """
        return f"Node at {self.position.tolist()}"

    @staticmethod
    def add_neighbor(self, neighbor: "Node"):
        """
        Add a neighboring node to this node's list of neighbors using a cache mechanism to ensure efficient memory usage.

        Args:
            neighbor (Node): The neighbor node to add.
        """
        if neighbor.id not in self.neighbors:
            self.neighbors[neighbor.id] = neighbor
        logging.debug(
            f"Added neighbor {neighbor} to node at {self.position.tolist()} with efficient caching mechanism"
        )

    # Other Node methods...


# Grid Class
class Grid:
    def __init__(self, width, height, tile_size, pool):
        """
        Initialize a Grid instance with specified dimensions and tile size, optimized for high performance
        and efficient memory usage using advanced data structures. The multiprocessing pool is passed as an argument
        to avoid creating new pools in daemon processes, ensuring better resource management and performance.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            tile_size (int): The size of each tile in the grid.
            pool (multiprocessing.Pool): A multiprocessing pool shared across the application for concurrent operations.

        Attributes:
            width (int): Stores the width of the grid.
            height (int): Stores the height of the grid.
            tile_size (int): Stores the size of each tile.
            tiles (np.ndarray): A 2D array of Tile objects representing the grid, initialized using the provided pool.
        """
        self.pool = pool if pool is not None else GLOBAL_POOL
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.tiles = np.empty(
            (width, height), dtype=object
        )  # Pre-allocate space for Tile objects

        # Generate parameters for each tile
        tile_params = [
            (x, y, tile_size, (255, 255, 255))
            for x in range(width)
            for y in range(height)
        ]
        # Map tile initialization across the provided multiprocessing pool
        tiles_flat = pool.starmap(initialize_tile, tile_params)
        # Reshape the flat list of tiles into a 2D grid
        self.tiles = np.array(tiles_flat).reshape((width, height))


@staticmethod
def initialize_tile(x, y, tile_size, color):
    """
    Static method to initialize a Tile object, designed to be used with a multiprocessing pool for efficient
    parallel initialization of tiles.

    Args:
        x (int): The x-coordinate of the tile.
        y (int): The y-coordinate of the tile.
        tile_size (int): The size of the tile.
        color (Tuple[int, int, int]): The color of the tile.

    Returns:
        Tile: An initialized Tile object.
    """
    return Tile(x, y, tile_size, color)


# Tile Class
class Tile:
    def __init__(self, x, y, size, color, pool):
        """
        Initialize a Tile instance with position, size, and color, employing advanced data management and optimization techniques.

        Args:
            x (int): The x-coordinate of the tile.
            y (int): The y-coordinate of the tile.
            size (int): The size of the tile.
            color (Tuple[int, int, int]): The color of the tile.
            pool (multiprocessing.Pool): A multiprocessing pool shared across the application for concurrent operations.

        Attributes:
            x (int): Stores the x-coordinate of the tile.
            y (int): Stores the y-coordinate of the tile.
            size (int): Stores the size of the tile.
            color (np.ndarray): Stores the color of the tile, utilizing numpy for efficient array manipulation.
            process_pool (multiprocessing.Pool): Stores the shared multiprocessing pool for potential parallel processing enhancements.
        """
        self.x = x
        self.y = y
        self.size = size
        self.color = np.array(
            color, dtype=np.uint8
        )  # Using numpy array for efficient data handling and operations

        # Reference to the shared multiprocessing pool
        self.pool = pool if pool is not None else GLOBAL_POOL

        # Setting up caching for tile properties to enhance performance
        self.tile_cache = cachetools.LRUCache(maxsize=1024)

        # Asynchronous operations setup to ensure non-blocking behavior in high-load scenarios
        self.tile_lock = asyncio.Lock()

    @staticmethod
    async def update_tile_color(tile, new_color):
        """
        Asynchronously update the color of the tile, utilizing multiprocessing for performance optimization.

        Args:
            tile (Tile): The tile instance to be updated.
            new_color (Tuple[int, int, int]): The new color to be set for the tile.
        """
        async with tile.tile_lock:
            # Offload the color update task to the multiprocessing pool for performance optimization
            result = await asyncio.get_event_loop().run_in_executor(
                tile.process_pool, Tile._change_color, tile, new_color
            )
            tile.color = np.array(result, dtype=np.uint8)

    @staticmethod
    def _change_color(tile, new_color):
        """
        Change the color of the tile. This function is designed to run in a separate process for performance optimization.

        Args:
            tile (Tile): The tile instance to be updated.
            new_color (Tuple[int, int, int]): The new color to be set for the tile.

        Returns:
            Tuple[int, int, int]: The new color of the tile.
        """
        return new_color

    # Other Tile methods...


# Fruit Class
class Fruit:
    def __init__(self, position, size, color, border_color, border_thickness, pool):
        """
        Initialize a Fruit instance with position, size, color, border color, and border thickness, employing advanced
        data management and optimization techniques. Utilizes a shared multiprocessing pool passed from the main application
        to avoid creating new pools in subprocesses.

        Args:
            position (Tuple[int, int]): The (x, y) coordinates of the fruit on the grid.
            size (int): The size of the fruit.
            color (Tuple[int, int, int]): The color of the fruit.
            border_color (Tuple[int, int, int]): The color of the fruit's border.
            border_thickness (int): The thickness of the fruit's border.
            pool (multiprocessing.Pool): The shared multiprocessing pool.

        Attributes:
            position (np.ndarray): Stores the coordinates of the fruit as a numpy array for efficient computation.
            size (int): Stores the size of the fruit.
            color (np.ndarray): Stores the color of the fruit as a numpy array to facilitate rapid operations.
            border_color (np.ndarray): Stores the color of the fruit's border as a numpy array.
            border_thickness (int): Stores the thickness of the fruit's border.
            process_pool (multiprocessing.Pool): Stores the shared multiprocessing pool.
        """
        self.position = np.array(position, dtype=np.int32)
        self.size = size
        self.color = np.array(color, dtype=np.uint8)
        self.border_color = np.array(border_color, dtype=np.uint8)
        self.border_thickness = border_thickness
        self.process_pool = pool if pool is not None else GLOBAL_POOL

        # Implementing caching for rapid access to fruit properties
        self.properties_cache = cachetools.LRUCache(maxsize=128)

        # Asynchronous handling for updating fruit properties
        self.update_lock = asyncio.Lock()

    @staticmethod
    async def update_fruit_properties(
        fruit,
        new_position=None,
        new_size=None,
        new_color=None,
        new_border_color=None,
        new_border_thickness=None,
    ):
        """
        Asynchronously update the fruit properties, utilizing the shared multiprocessing pool for performance optimization.

        Args:
            fruit (Fruit): The fruit instance to be updated.
            new_position (Optional[Tuple[int, int]]): New position of the fruit.
            new_size (Optional[int]): New size of the fruit.
            new_color (Optional[Tuple[int, int, int]]): New color of the fruit.
            new_border_color (Optional[Tuple[int, int, int]]): New border color of the fruit.
            new_border_thickness (Optional[int]): New border thickness of the fruit.
        """
        async with fruit.update_lock:
            if new_position is not None:
                fruit.position = np.array(new_position, dtype=np.int32)
            if new_size is not None:
                fruit.size = new_size
            if new_color is not None:
                fruit.color = np.array(new_color, dtype=np.uint8)
            if new_border_color is not None:
                fruit.border_color = np.array(new_border_color, dtype=np.uint8)
            if new_border_thickness is not None:
                fruit.border_thickness = new_border_thickness

            # Offload the computation-intensive task to the shared multiprocessing pool
            result = await asyncio.get_event_loop().run_in_executor(
                fruit.process_pool, fruit.cache_updated_properties
            )

    @staticmethod
    def cache_updated_properties(fruit):
        """
        Cache the updated properties of the fruit to ensure rapid access during game rendering and logic computation.
        """
        fruit.properties_cache["position"] = fruit.position
        fruit.properties_cache["size"] = fruit.size
        fruit.properties_cache["color"] = fruit.color
        fruit.properties_cache["border_color"] = fruit.border_color
        fruit.properties_cache["border_thickness"] = fruit.border_thickness

    # Other Fruit methods...


# Snake Class
class Snake:
    def __init__(self, segments, color, pool):
        """
        Initialize a Snake instance with segments, color, and a shared multiprocessing pool, employing advanced data management and optimization techniques.

        Args:
            segments (Deque[Node]): A deque of Node objects representing the snake's body segments.
            color (Tuple[int, int, int]): The color of the snake.
            process_pool (multiprocessing.Pool): A shared multiprocessing pool for concurrent operations.

        Attributes:
            segments (Deque[Node]): Stores the snake's body segments, optimized for rapid access and mutation.
            color (np.ndarray): Stores the color of the snake, utilizing numpy for efficient array manipulation.
            segment_positions (np.ndarray): Cached positions of the snake's segments for quick access.
            segment_directions (np.ndarray): Cached directions of each segment for dynamic movement calculations.
            lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during updates in asynchronous environments.
            process_pool (multiprocessing.Pool): A shared multiprocessing pool to avoid creating new pools in daemon processes.
        """
        self.segments = segments
        self.color = np.array(color, dtype=np.uint8)
        self.segment_positions = np.array(
            [node.position for node in segments], dtype=np.int32
        )
        self.segment_directions = np.zeros(
            (len(segments), 2), dtype=np.int32
        )  # Placeholder for actual direction data
        self.lock = asyncio.Lock()
        self.pool = pool if pool is not None else GLOBAL_POOL

        # Setup caching for segment data to enhance performance
        self.segments_cache = cachetools.LRUCache(maxsize=1024)

    async def update_segment_positions(self):
        """
        Asynchronously update the positions of the snake's segments, utilizing the shared multiprocessing pool for performance optimization.

        This method recalculates the positions based on movement and updates the cached positions.
        """
        async with self.lock:
            # Offload the computation-intensive task to the shared multiprocessing pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.process_pool, Snake.calculate_new_positions, self
            )
            self.segment_positions = np.array(result, dtype=np.int32)

    @staticmethod
    def calculate_new_positions(snake):
        """
        Calculate new positions for each segment based on current directions and positions.
        This function is designed to run in a separate process for performance optimization using the shared multiprocessing pool.

        Args:
            snake (Snake): The instance of Snake for which to calculate new positions.

        Returns:
            List[Tuple[int, int]]: List of new positions for each segment.
        """
        new_positions = []
        for idx, pos in enumerate(snake.segment_positions):
            direction = snake.segment_directions[idx]
            new_position = (pos[0] + direction[0], pos[1] + direction[1])
            new_positions.append(new_position)
        return new_positions

    # Other Snake methods...


# Pathfinding Class
class Pathfinding:
    def __init__(self, grid, pool):
        """
        Initialize the Pathfinding class which is responsible for calculating optimal paths for the snake using advanced algorithms.

        Args:
            grid (Grid): The game grid which contains all the necessary information about the game state.
            pathfinding_pool (mp.Pool): A multiprocessing pool to handle concurrent pathfinding computations, ensuring that no new pools are created in daemon processes.

        Attributes:
            grid (Grid): Stores the reference to the game grid.
            path_cache (LRUCache): Caches the results of complex path calculations for quick retrieval.
            pathfinding_pool (mp.Pool): A shared multiprocessing pool to handle concurrent pathfinding computations.
            pathfinding_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during path calculations.
        """
        self.grid = grid
        self.path_cache = LRUCache(maxsize=1024)
        self.pool = pool if pool is not None else GLOBAL_POOL
        self.pathfinding_lock = asyncio.Lock()

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
            # Check if the path is already in the cache
            path_key = (hash(start), hash(goal))
            if path_key in self.path_cache:
                return self.path_cache[path_key]

            # Offload the computation-intensive task to the shared multiprocessing pool
            path = await asyncio.get_event_loop().run_in_executor(
                self.pool, self._a_star_search, start, goal
            )
            # Cache the result of the path computation
            self.path_cache[path_key] = path
            return path

    @staticmethod
    def _a_star_search(start, goal, grid):
        """
        Implements the A* search algorithm to find the shortest path from start to goal.

        Args:
            start (Node): The starting node.
            goal (Node): The goal node.
            grid (Grid): The grid instance to access neighbors and distances.

        Returns:
            List[Node]: The path from start to goal as a list of nodes.
        """
        open_set = deque([start])
        came_from = {}
        g_score = {start: 0}
        f_score = {start: grid._heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda o: f_score[o])
            if current == goal:
                return grid._reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in grid.get_neighbors(current):
                tentative_g_score = g_score[current] + grid.distance(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + grid._heuristic(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return []

    @staticmethod
    def _heuristic(node1, node2):
        """
        Calculate the heuristic estimated cost from node1 to node2 using Manhattan distance.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            float: The estimated cost from node1 to node2.
        """
        return np.linalg.norm(np.array(node1.position) - np.array(node2.position))

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
        return total_path[::-1]  # Return reversed path


# Perception Class
class Perception:
    def __init__(self, grid, pool):
        """
        Initialize the Perception class which is responsible for evaluating and updating the game's perception of the environment.

        Args:
            grid (Grid): The game grid which contains all the necessary information about the game state.
            pool (multiprocessing.Pool): A multiprocessing pool shared across the application for concurrent operations.

        Attributes:
            grid (Grid): Stores the reference to the game grid.
            perception_cache (LRUCache): Caches the results of complex perception calculations for quick retrieval.
            perception_update_pool (mp.Pool): A multiprocessing pool to handle concurrent perception updates.
            perception_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during updates.
        """
        self.grid = grid
        self.perception_cache = LRUCache(maxsize=1024)
        self.pool = pool if pool is not None else GLOBAL_POOL
        self.perception_lock = asyncio.Lock()

    async def update_perception(self, snake, fruit):
        """
        Asynchronously updates the perception of the environment based on the current state of the snake and the fruit.

        Args:
            snake (Snake): The current state of the snake.
            fruit (Fruit): The current state of the fruit.

        Returns:
            None
        """
        async with self.perception_lock:
            # Offload the computation-intensive task to the multiprocessing pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.pool, self._compute_perception, snake, fruit
            )
            # Cache the result of the perception computation
            self.perception_cache[hash((snake, fruit))] = result

    @staticmethod
    def _compute_perception(snake, fruit):
        """
        Compute the perception details such as potential collisions, pathfinding evaluations, and strategic positioning.

        Args:
            snake (Snake): The current state of the snake.
            fruit (Fruit): The current state of the fruit.

        Returns:
            Dict[str, Any]: A dictionary containing detailed perception metrics.
        """
        # Example of a complex computation (placeholder for actual implementation)
        path = np.linalg.norm(
            np.array(snake.get_head_position()) - np.array(fruit.get_position())
        )
        return {
            "path_length": path,
            "collision_risk": Perception._evaluate_collision_risk(snake, path),
            "strategic_advantage": Perception._calculate_strategic_advantage(
                snake, fruit
            ),
        }

    @staticmethod
    def _evaluate_collision_risk(snake, path_length):
        """
        Evaluate the risk of collision based on the path length and snake's current trajectory.

        Args:
            snake (Snake): The current state of the snake.
            path_length (float): The computed path length from the snake's head to the fruit.

        Returns:
            float: A risk factor indicating the likelihood of collision.
        """
        # Placeholder logic for collision risk
        return path_length / (1 + len(snake.get_segments()))

    @staticmethod
    def _calculate_strategic_advantage(snake, fruit):
        """
        Calculate the strategic advantage of moving towards the fruit based on current game metrics.

        Args:
            snake (Snake): The current state of the snake.
            fruit (Fruit): The current state of the fruit.

        Returns:
            float: A score representing the strategic advantage.
        """
        # Placeholder logic for strategic advantage
        return np.random.random()


# Decision Class
class Decision:
    def __init__(self, pool):
        """
        Initialize the Decision class which is responsible for making strategic decisions based on the game state.

        Attributes:
            decision_cache (LRUCache): A cache to store the results of complex decision-making computations for quick retrieval.
            decision_pool (multiprocessing.Pool): A pool of worker processes to handle decision-making tasks concurrently, passed from the main application to ensure a single pool instance.
            decision_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during decision-making processes.
            logger (logging.Logger): A logger to record all decision-making activities with detailed information.
        """
        self.decision_cache = LRUCache(maxsize=1024)
        self.pool = pool if pool is not None else GLOBAL_POOL
        self.decision_lock = asyncio.Lock()
        self.logger = logging.getLogger("Decision")
        self._setup_logger()

    def _setup_logger(self):
        """
        Setup the logging configuration for the Decision class to ensure all activities are logged with detailed information.

        This method configures a StreamHandler with a specific format for logging messages, which includes the timestamp,
        logger name, log level, and the log message. The log level is set to DEBUG to capture detailed information for
        troubleshooting and analysis.
        """
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    async def make_decision(self, game_state):
        """
        Asynchronously make a decision based on the current game state.

        This method first attempts to retrieve a cached decision using a hash of the game state. If the decision is not
        in the cache, it computes a new decision by offloading the computation to a separate process in the decision pool.
        The new decision is then cached and logged.

        Args:
            game_state (dict): The current state of the game, represented as a dictionary.

        Returns:
            Any: The decision made based on the game state, which could be a direction for the snake to move or other strategic actions.
        """
        async with self.decision_lock:
            game_state_hash = hash(frozenset(game_state.items()))
            if game_state_hash in self.decision_cache:
                self.logger.debug("Decision retrieved from cache.")
                return self.decision_cache[game_state_hash]

            decision = await asyncio.get_event_loop().run_in_executor(
                self.decision_pool, self._compute_decision, game_state
            )
            self.decision_cache[game_state_hash] = decision
            self.logger.debug(f"Decision computed and cached: {decision}")
            return decision

    def _compute_decision(self, game_state):
        """
        Compute a decision based on the game state. This method is intended to be run in a separate process to utilize
        multiprocessing capabilities for performance optimization.

        The decision-making logic is currently implemented as a placeholder using a random choice among possible moves.
        This method should be replaced with a more sophisticated AI algorithm in future implementations.

        Args:
            game_state (dict): The current state of the game.

        Returns:
            Any: The decision computed from the game state, typically a move direction.
        """
        decision = np.random.choice(["move_left", "move_right", "move_up", "move_down"])
        return decision

    # Other Decision methods...


# Renderer Class
class Renderer:
    def __init__(self, surface: pygame.Surface, pool: mp.Pool):
        """
        Initialize the Renderer class which is responsible for all graphical rendering in the game, utilizing a shared multiprocessing pool to handle concurrent rendering tasks efficiently.

        Args:
            surface (pygame.Surface): The main surface on which all graphical elements are drawn.
            pool (multiprocessing.Pool): A shared multiprocessing pool for handling concurrent rendering tasks.

        Attributes:
            surface (pygame.Surface): Stores the main drawing surface.
            cache (cachetools.LRUCache): A cache for storing pre-rendered images to improve rendering performance.
            pool (multiprocessing.Pool): A pool of worker processes to handle rendering tasks concurrently, passed from the main application to ensure efficient resource management.
        """
        self.surface = surface
        self.cache = cachetools.LRUCache(maxsize=1024)
        self.pool = pool if pool is not None else GLOBAL_POOL

    def render_scene(self, game_objects: list):
        """
        Render all game objects onto the main surface using the shared multiprocessing pool to manage the rendering tasks concurrently.

        Args:
            game_objects (list): A list of game objects which have a draw method.
        """
        render_tasks = [
            self.pool.apply_async(obj.draw, (self.surface,)) for obj in game_objects
        ]
        for task in render_tasks:
            task.get()

    def apply_strobe_effects(self, frequency: float):
        """
        Apply strobe lighting effects to the entire game scene using asynchronous methods to enhance game visuals without blocking the main game loop.

        Args:
            frequency (float): The frequency of the strobe effect in Hz.
        """
        asyncio.run(self._async_strobe_effect(frequency))

    @staticmethod
    async def _async_strobe_effect(frequency: float):
        """
        Asynchronously apply strobe effects at the specified frequency to enhance game visuals, utilizing asyncio to manage the timing without blocking.

        Args:
            frequency (float): The frequency of the strobe effect in Hz.
        """
        interval = 1 / frequency
        while True:
            await asyncio.sleep(interval)
            pygame.display.get_surface().fill((255, 255, 255))
            await asyncio.sleep(interval)
            pygame.display.get_surface().fill((0, 0, 0))

    def apply_rainbow_effects(self, game_objects: list):
        """
        Apply rainbow color effects to specified game objects using the shared multiprocessing pool to handle the color change tasks concurrently.

        Args:
            game_objects (list): A list of game objects which can change color.
        """
        colors = [
            (255, 0, 0),
            (255, 165, 0),
            (255, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
            (75, 0, 130),
            (238, 130, 238),
        ]
        for obj in game_objects:
            color_tasks = [
                self.pool.apply_async(obj.set_color, (color,)) for color in colors
            ]
            for task in color_tasks:
                task.get()
                obj.draw(self.surface)
                pygame.display.flip()

    def update_display(self):
        """
        Update the display after all rendering tasks are complete, ensuring that the visual updates are reflected on the screen.
        """
        pygame.display.flip()

    def clear_cache(self):
        """
        Clear the rendering cache to free up memory, ensuring optimal performance by removing outdated or unused pre-rendered images.
        """
        self.cache.clear()


# GUI Class
class GUI:
    def __init__(
        self,
        screen_dimensions: Tuple[int, int],
        font_size: int = 20,
        pool: Optional[mp.Pool] = None,
    ):
        """
        Initialize the GUI system for the Snake game, meticulously setting up the screen, font, and caching mechanisms to ensure optimal performance and user experience. This initialization now includes the option to pass an existing multiprocessing pool to be used for potential parallel processing tasks within the GUI operations.

        Args:
            screen_dimensions (Tuple[int, int]): The width and height of the screen, specified in pixels, to define the area available for rendering the game's graphical content.
            font_size (int): The size of the font for text rendering, specified in points, which determines the visual clarity and readability of text displayed on the screen.
            pool (Optional[mp.Pool]): An optional multiprocessing pool that can be passed to the GUI for use in parallel processing tasks. If not provided, the GUI will operate without parallel processing enhancements.

        Attributes:
            screen (pygame.Surface): The main screen surface for the game, which acts as the canvas where all graphical elements are drawn. This surface is initialized based on the provided screen dimensions.
            font (pygame.font.Font): Font used for rendering text, initialized with a default system font unless specified otherwise, and set to the provided font size to ensure text is legible.
            cache (cachetools.LRUCache): A least recently used (LRU) cache for storing pre-rendered text surfaces to improve rendering performance by avoiding redundant rendering operations for the same text content. The cache is set with a maximum size of 100 entries, balancing memory usage and performance.
            pool (Optional[mp.Pool]): Stores the optional multiprocessing pool passed during initialization for use in potential parallel processing tasks within the GUI.
        """
        pygame.init()
        self.screen = pygame.display.set_mode(screen_dimensions)
        self.font = pygame.font.Font(None, font_size)
        self.cache = cachetools.LRUCache(
            maxsize=100
        )  # Initialize cache with a maximum of 100 text surfaces
        self.pool = pool if pool is not None else GLOBAL_POOL

    @staticmethod
    async def draw_text(
        gui_instance,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Asynchronously draw text on the screen at the specified position, utilizing caching and optional parallel processing to enhance performance significantly. This method handles the drawing operations in a non-blocking manner using asynchronous programming techniques, and may utilize the multiprocessing pool if available for parallel text rendering tasks.

        Args:
            gui_instance: The instance of the GUI class.
            text (str): The text to be rendered, which is a string of characters. This text is rendered onto the screen at the specified position and in the specified color.
            position (Tuple[int, int]): The (x, y) coordinates on the screen where the text will be drawn. The coordinates specify the top-left corner of the text surface.
            color (Tuple[int, int, int]): The RGB color of the text, specified as a tuple of three integers ranging from 0 to 255. This color determines how the text appears against the background.

        Detailed Operations:
            - Check if the text is already in the cache:
                - If not, render the text into a surface and store this surface in the cache.
                - If it is in the cache, retrieve the pre-rendered surface.
            - If a multiprocessing pool is available, offload the rendering task to the pool for enhanced performance.
            - Perform the blitting (bit-block transfer) of the text surface onto the main screen surface at the specified position.
            - This operation is performed asynchronously to ensure that the GUI remains responsive and performant even during intensive rendering operations.
        """
        if text not in gui_instance.cache:
            if gui_instance.pool:
                # Utilize the multiprocessing pool to render the text asynchronously
                text_surface = await asyncio.get_event_loop().run_in_executor(
                    gui_instance.pool, gui_instance.font.render, text, True, color
                )
            else:
                # Render the text without multiprocessing enhancements
                text_surface = gui_instance.font.render(text, True, color)
            gui_instance.cache[text] = text_surface
        else:
            # Retrieve the text surface from the cache
            text_surface = gui_instance.cache[text]

        # Asynchronously blit the text surface to the screen
        await asyncio.to_thread(gui_instance.screen.blit, text_surface, position)

    def update_display(self):
        """
        Update the display to reflect the latest changes, ensuring all drawn elements are visible. This method is crucial for maintaining the visual integrity of the game's interface after any graphical updates.

        Detailed Operations:
            - Flip the display: This operation updates the actual display seen by the user with all the changes made to the screen surface during the current frame. It is equivalent to refreshing the screen to show the latest graphical content.
        """
        pygame.display.flip()

    def clear_screen(self, color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Clear the screen with a uniform color to prepare for the next frame, ensuring a clean visual state before any new drawing operations are performed.

        Args:
            color (Tuple[int, int, int]): The RGB color to fill the screen, specified as a tuple of three integers ranging from 0 to 255. This color becomes the background color of the screen until the next clearing operation.

        Detailed Operations:
            - Fill the screen: The entire screen surface is filled with the specified color, effectively clearing any previous graphical content and setting a uniform background color for new drawings.
        """
        self.screen.fill(color)


# Game Logic Class
class GameLogic:
    def __init__(self, pool):
        """
        Initializes the GameLogic class which is responsible for managing the core game loop,
        handling events, updating game states, and interfacing with other components like
        Pathfinding, Decision, and Renderer to ensure a seamless and efficient gameplay experience.

        Args:
            pool (multiprocessing.Pool): A multiprocessing pool shared across the application for concurrent operations.
        """
        self.event_queue = asyncio.Queue()
        self.game_state = {}
        self.pathfinding = Pathfinding(
            grid=Grid(width=100, height=100, tile_size=10, pool=pool)
        )
        self.decision = Decision()
        self.renderer = Renderer()
        self.gui = GUI()
        self.updater = Updater()
        self.running = False
        self.game_speed = 1.0  # Default game speed multiplier
        self.cache = cachetools.LRUCache(maxsize=1024)

        # Use the passed multiprocessing pool
        self.pool = pool if pool is not None else GLOBAL_POOL

        # Setup logging
        self.logger = logging.getLogger("GameLogic")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("game_logic.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def handle_events(self):
        """
        Asynchronously handle game events such as user inputs, system events, and other interactions,
        ensuring that the game responds in real-time without delays.
        """
        while self.running:
            event = self.event_queue.get_nowait()
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.process_event(event)

    @staticmethod
    async def process_event(event):
        """
        Process individual events using asynchronous processing to ensure non-blocking operations
        which can handle complex computations like AI decisions or pathfinding without freezing the UI.
        """
        if event.key == pygame.K_UP:
            await Decision.make_decision("UP")
        elif event.key == pygame.K_DOWN:
            await Decision.make_decision("DOWN")
        elif event.key == pygame.K_LEFT:
            await Decision.make_decision("LEFT")
        elif event.key == pygame.K_RIGHT:
            await Decision.make_decision("RIGHT")

    @classmethod
    def update_game_state(cls):
        """
        Update the game state by interfacing with the Pathfinding, Decision, and Renderer components,
        utilizing multiprocessing to handle complex calculations and state updates concurrently.
        """
        cls.shared_state["snake"] = Pathfinding.move_snake(
            cls.shared_state.get("snake", None)
        )
        decision_future = cls.process_pool.apply_async(
            Decision.integrate_data, (Pathfinding, cls.shared_state)
        )
        cls.game_state["decision"] = decision_future.get()

        # Cache the current game state for quick retrieval
        state_hash = hashlib.sha256(json.dumps(cls.game_state).encode()).hexdigest()
        cls.cache[state_hash] = cls.game_state

    def start_game_loop(self):
        """
        Start the main game loop, managing all game processes and ensuring high performance and responsiveness
        through asynchronous operations and effective multiprocessing.
        """
        self.running = True
        asyncio.run(self.main_async_loop())


# Updater Class
class Updater:
    def __init__(self, pool):
        """
        Initializes the Updater class which is responsible for managing updates across all components of the game,
        handling errors, logging activities, and ensuring a graceful exit. This class utilizes a shared multiprocessing
        pool for handling concurrent updates, caching for efficient data retrieval, and asynchronous operations to enhance
        performance and responsiveness.

        Args:
            pool (multiprocessing.Pool): A shared multiprocessing pool for concurrent operations.
        """
        self.logger = logging.getLogger(__name__)
        self.update_queue = mp.Queue()
        self.cache = cachetools.LRUCache(maxsize=1024)
        self.lock = threading.Lock()
        self.pool = pool if pool is not None else GLOBAL_POOL
        self.event_loop = asyncio.get_event_loop()
        self.tasks = []
        updater = self.logger

    @staticmethod
    def update_all(updater):
        """
        Asynchronously and concurrently updates all game components, ensuring high performance and efficiency.
        Utilizes the shared multiprocessing pool to handle complex computations and caching to retrieve pre-computed results rapidly.
        """
        try:
            with updater.lock:
                while not updater.update_queue.empty():
                    component = updater.update_queue.get()
                    if component in updater.cache:
                        continue
                    task = updater.pool.apply_async(component.update)
                    updater.tasks.append(task)
                    updater.cache[component] = task.get()
        except Exception as e:
            updater.logger.error(f"Error during update: {e}")
            updater.handle_errors(e)

    @staticmethod
    def logger(updater, message: str):
        """
        Logs significant game activities or errors using detailed logging.
        """
        updater.logger.info(message)

    @staticmethod
    def handle_errors(updater, error):
        """
        Handles any errors that occur during the game's execution, logs them, and attempts recovery if possible.
        """
        updater.logger.error(f"Encountered error: {error}")
        # Attempt recovery or escalate
        if not Updater.recover_from_error(updater, error):
            raise RuntimeError("Critical error, unable to recover.")

    @staticmethod
    def ensure_graceful_exit(updater):
        """
        Ensures a graceful exit from the game, handling any cleanup or final logging needed.
        """
        updater.logger.info("Initiating graceful exit.")
        GlobalPool.close_pool()
        updater.event_loop.close()
        updater.logger.info("Graceful exit completed.")

    @staticmethod
    def recover_from_error(updater, error):
        """
        Attempts to recover from an error, returning True if successful.
        """
        try:
            # Placeholder for recovery logic
            updater.logger.info("Attempting to recover from error.")
            return True
        except Exception as e:
            updater.logger.error(f"Recovery failed: {e}")
            return False


# Main function
class GameExecution:
    """
    This class orchestrates the initialization, execution, and termination of the game loop,
    managing all game components and ensuring a seamless and robust gameplay experience.
    """

    @staticmethod
    def main(pool=GlobalPool.get_pool()):
        """
        The main function orchestrates the initialization, execution, and termination of the game loop,
        managing all game components and ensuring a seamless and robust gameplay experience.
        """
        # Initialize multiprocessing pool with the number of available CPU cores
        pool = pool if pool is not None else GLOBAL_POOL
        try:
            # Initialize all game components with the multiprocessing pool
            game_logic = GameLogic(
                pool
            )  # Manages the core game logic including event handling and state updates
            renderer = Renderer(
                pool
            )  # Handles all rendering tasks to display game elements
            updater = Updater(
                pool
            )  # Manages updates and logging across various game components
            gui = GUI(pool)  # Manages the graphical user interface elements
            gui.setup_elements()  # Setup GUI elements such as buttons, scoreboards, etc.

            # Initialize Pygame and set up the game environment
            pygame.init()
            clock = pygame.time.Clock()
            fps = 60  # Frames per second, defining the refresh rate of the game loop

            # Start the game loop using GameLogic's start_game_loop method
            GameLogic.start_game_loop(game_logic, renderer, updater, gui, clock, fps)

        finally:
            # Ensure a graceful exit by performing necessary cleanup
            Updater.ensure_graceful_exit(
                Updater
            )  # Ensure a graceful exit by performing necessary cleanup
            if GlobalPool.get_pool() is not None:
                GlobalPool.close_pool()  # Close the multiprocessing pool


# Entry point
if __name__ == "__main__":
    GameExecution.main()
