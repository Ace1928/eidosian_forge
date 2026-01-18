"""
Module: 2048 Game AI - Player and Tile Management
Author: [Author Name]
Version: 1.0.0
Purpose: This module is designed to define and manage the Tile and Player classes for a 2048 game AI. It incorporates advanced logging, error handling, and method decorators to enhance functionality, maintainability, and readability. The module aims to provide a robust framework for simulating and analyzing player strategies in the 2048 game environment.
Dependencies:
- numpy: Utilized for array manipulation and mathematical operations, essential for managing the game state.
- random: Employed for generating random numbers, crucial for tile placement and value assignment.
- logging: Integrated for detailed logging of method calls, parameter values, and error handling, facilitating debugging and maintenance.
"""

# Importing necessary libraries
import numpy as np
import random
from typing import List, Optional, Tuple
import logging
import time  # Imported for measuring function execution time

# Configuring logging to debug level with a specific format for timestamps, log level, and messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Decorator for logging and error handling
def log_and_handle_errors(func):
    """
    A decorator that wraps the passed-in function, logs method calls with their arguments, and handles exceptions by logging them. This enhances the debuggability of the code by providing insights into function execution and error occurrences.

    Additionally, it measures and logs the execution time of the function, offering insights into performance aspects.

    Parameters:
    - func (Callable): The function to wrap.

    Returns:
    - Callable: The wrapped function with added logging, error handling, and execution time measurement capabilities.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time of function execution
        try:
            # Logging the function call with arguments and keyword arguments
            logging.debug(f"Calling {func.__name__} with {args} and {kwargs}")
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Logging any exceptions that occur during the function execution
            logging.error(f"Error occurred in {func.__name__}: {e}")
            # Reraising the exception after logging it
            raise e
        finally:
            end_time = time.time()  # End time of function execution
            # Logging the execution time of the function
            logging.debug(
                f"{func.__name__} executed in {end_time - start_time:.4f} seconds"
            )

    return wrapper


# Class definition for Tile
class Tile:
    """
    Represents a tile in the 2048 game with attributes for position, value, and state indicators for merging and movement. This class encapsulates the properties and behaviors of individual tiles on the game board.

    Attributes:
        position (np.array): The (x, y) position of the tile on the game board, represented as a numpy array for efficient manipulation.
        value (int): The numerical value of the tile, determining its strength in the game.
        death_on_impact (bool): A flag indicating if the tile will be merged (and thus removed) on the next move. This is used to manage tile merging logic.
        already_increased (bool): A flag to prevent a tile from merging more than once per move, ensuring correct game mechanics.
    """

    @log_and_handle_errors
    def __init__(self, x: int, y: int, value: int = 2) -> None:
        """
        Initializes a Tile instance with specified position and value. It also sets the initial state for merging and movement indicators.

        Parameters:
            x (int): The x-coordinate of the tile's position on the game board.
            y (int): The y-coordinate of the tile's position on the game board.
            value (int, optional): The value of the tile, determining its strength. Defaults to 2, following the game's rules.
        """
        # Initializing tile attributes
        self.position = np.array(
            [x, y], dtype=int
        )  # Position represented as a numpy array for efficient manipulation
        self.value = value  # Tile value
        self.death_on_impact = False  # Flag for merge removal
        self.already_increased = False  # Flag for merge prevention
        # Logging the creation of a new tile with its position and value
        logging.debug(
            f"Tile created at position {self.position} with value {self.value}"
        )

    @log_and_handle_errors
    def move_to(self, new_position: np.array) -> None:
        """
        Updates the tile's position to a new specified location. This method is called during tile movement operations in the game.

        Parameters:
            new_position (np.array): The new position to move the tile to, represented as a numpy array.
        """
        # Logging the movement operation from the current position to the new position
        logging.debug(f"Moving tile from {self.position} to {new_position}")
        # Updating the tile's position
        self.position = new_position
        # Logging the successful update of the tile's position
        logging.debug(f"Tile successfully moved to {new_position}")

    @log_and_handle_errors
    def set_color(self) -> None:
        """
        Placeholder method to update the tile's color based on its value. This method should be implemented in a UI-specific manner to visually distinguish tiles of different values.

        Note: This method currently does not perform any operations and serves as a placeholder for future UI integration.
        """
        # Logging the call to the placeholder method set_color
        logging.debug("Called set_color method on tile")
        pass  # Placeholder for UI integration

    @log_and_handle_errors
    def show(self) -> None:
        """
        Placeholder method to display the tile. This method should be implemented in a UI-specific manner to visually represent the tile on the game board.

        Note: This method currently does not perform any operations and serves as a placeholder for future UI integration.
        """
        # Logging the call to the placeholder method show
        logging.debug("Called show method on tile")
        pass  # Placeholder for UI integration

    @log_and_handle_errors
    def clone(self) -> "Tile":
        """
        Creates a copy of the tile, preserving its position and value. This method is useful for operations requiring tile duplication without altering the original tile.

        Returns:
            Tile: A new Tile instance with the same position and value as the original.
        """
        # Returning a new Tile instance with the same position and value
        return Tile(self.position[0], self.position[1], self.value)


# Class definition for Player
class Player:
    """
    Manages the state and behavior of a 2048 game player, including tile positions, movements, and score. This class encapsulates the logic for player actions, tile management, and game state updates.

    Attributes:
        fitness (int): A metric for evaluating the player's performance, useful in AI simulations and strategy analysis.
        dead (bool): Indicates whether the player has lost the game, based on the game's rules and conditions.
        score (int): The player's current score, calculated based on tile mergers and movements.
        tiles (List[Tile]): A list of Tile objects representing the tiles on the board, managed by the player.
        empty_positions (List[np.array]): A list of positions on the board that do not contain a tile, used for new tile placement.
        move_direction (np.array): The current direction in which the tiles are moving, represented as a numpy array.
        moving_tiles (bool): Flag indicating whether the tiles are currently moving, used to manage game state updates.
        tile_moved (bool): Flag indicating whether at least one tile has moved during the last move, used to determine game progression.
        starting_positions (np.array): The starting positions and values of the initial two tiles, used for game initialization and replays.
    """

    def __init__(self, is_replay: bool = False) -> None:
        """
        Initializes a Player instance with default attributes. It sets up the initial game state, including tile positions and score, and optionally prepares the game for a replay.

        Parameters:
            is_replay (bool, optional): Indicates whether this player instance is for a replay, affecting the initial game setup. Defaults to False.
        """
        # Initializing player attributes
        self.fitness: int = 0  # Player performance metric
        self.dead: bool = False  # Game loss indicator
        self.score: int = 0  # Current score
        self.tiles: List[Tile] = []  # List of tiles on the board
        self.empty_positions: List[np.array] = (
            []
        )  # List of empty positions on the board
        self.move_direction: np.array = np.array(
            [0, 0], dtype=int
        )  # Current tile movement direction
        self.moving_tiles: bool = False  # Tile movement flag
        self.tile_moved: bool = False  # Tile movement indicator
        self.starting_positions: np.array = np.zeros(
            (2, 3), dtype=int
        )  # Starting positions and values of initial tiles

        # Filling the list of empty positions on the board
        self.fill_empty_positions()
        # Adding initial tiles to the board if not a replay
        if not is_replay:
            self.add_new_tile()
            self.add_new_tile()
            # Setting the starting positions of the initial tiles
            self.set_starting_positions()
        # Logging the initialization of a new player instance
        logging.debug("Player initialized")

    def fill_empty_positions(self) -> None:
        self.empty_positions = [np.array([i, j]) for i in range(4) for j in range(4)]

    def set_empty_positions(self) -> None:
        self.empty_positions.clear()
        for i in range(4):
            for j in range(4):
                if self.get_value(i, j) == 0:
                    self.empty_positions.append(np.array([i, j]))

    def set_starting_positions(self) -> None:
        if len(self.tiles) >= 2:
            self.starting_positions[0, :] = np.append(
                self.tiles[0].position, self.tiles[0].value
            )
            self.starting_positions[1, :] = np.append(
                self.tiles[1].position, self.tiles[1].value
            )

    def add_new_tile(self, value: Optional[int] = None) -> None:
        if not self.empty_positions:
            return
        index = random.randint(0, len(self.empty_positions) - 1)
        position = self.empty_positions.pop(index)
        if value is None:
            value = 4 if random.random() < 0.1 else 2
        new_tile = Tile(position[0], position[1], value)
        new_tile.set_color()
        self.tiles.append(new_tile)

    def add_new_tile(self, value: Optional[int] = None) -> None:
        """
        Adds a new tile to the game board at a random empty position.

        Parameters:
            value (Optional[int]): The value of the new tile. If None, the value is randomly set to 2 or 4.
        if not self.empty_positions:
            logging.warning("No empty positions available to add a new tile.")
            return

        try:
            index = random.randint(0, len(self.empty_positions) - 1)
            position = self.empty_positions.pop(index)
            if value is None:
                value = 4 if random.random() < 0.1 else 2
            new_tile = Tile(position[0], position[1], value)
            new_tile.set_color()  # Placeholder for UI integration
            self.tiles.append(new_tile)
            logging.info(f"Added new tile at {position} with value {value}")
        except Exception as e:
            logging.error(f"Failed to add a new tile: {e}")
        """

    def show(self) -> None:
        """
        Iterates through the tiles, sorts them based on the death_on_impact flag, and calls the show method on each tile. This method is responsible for visually representing the current state of the game board.
        """
        for tile in sorted(self.tiles, key=lambda x: x.death_on_impact):
            tile.show()
        # Logging the completion of the show operation for all tiles
        logging.debug("Completed showing all tiles")

    def move_tiles(self) -> None:
        """
        Moves the tiles in the direction specified by `move_direction` and handles merging of tiles. This method is a key part of the game's logic, determining how tiles interact with each other and the game board.
        """
        self.tile_moved = False
        for tile in self.tiles:
            tile.already_increased = False

        if np.any(self.move_direction != 0):
            sorting_order = self.calculate_sorting_order()
            for order in sorting_order:
                for tile in self.tiles:
                    if np.array_equal(tile.position, order):
                        self.process_tile_movement(tile)
            if self.tile_moved:
                logging.info("Tiles moved")
            else:
                logging.info("No tiles moved")
        else:
            logging.debug("Move direction is zero; no tiles moved")

    def calculate_sorting_order(self) -> List[np.array]:
        sorting_vec = (
            np.array([3, 0])
            if self.move_direction[0] == 1
            else (
                np.array([0, 0])
                if self.move_direction[0] == -1
                else (
                    np.array([0, 3])
                    if self.move_direction[1] == 1
                    else np.array([0, 0])
                )
            )
        )
        vert = self.move_direction[1] != 0
        sorting_order = []
        for i in range(4):
            for j in range(4):
                temp = sorting_vec.copy()
                if vert:
                    temp[0] += j
                else:
                    temp[1] += j
                sorting_order.append(temp)
            sorting_vec -= self.move_direction
        return sorting_order

    def process_tile_movement(self, tile: Tile) -> None:
        move_to = tile.position + self.move_direction
        while self.is_position_empty(move_to):
            tile.move_to(move_to)
            move_to += self.move_direction
            self.tile_moved = True
        self.handle_potential_merge(tile, move_to)

    def is_position_empty(self, position: np.array) -> bool:
        return all(not np.array_equal(t.position, position) for t in self.tiles)

    def handle_potential_merge(self, tile: Tile, position: np.array) -> None:
        other = self.get_tile_at(position)
        if other and other.value == tile.value and not other.already_increased:
            tile.move_to(position)
            tile.death_on_impact = True
            other.already_increased = True
            other.value *= 2
            self.score += other.value
            other.set_color()
            self.tile_moved = True

    def get_tile_at(self, position: np.array) -> Optional[Tile]:
        for tile in self.tiles:
            if np.array_equal(tile.position, position):
                return tile
        return None

    def get_value(self, x: int, y: int) -> int:
        tile = self.get_tile_at(np.array([x, y]))
        return tile.value if tile else 0

    def move(self) -> None:
        if self.moving_tiles:
            for tile in self.tiles:
                tile.position += self.move_direction
            if self.done_moving():
                self.tiles = [tile for tile in self.tiles if not tile.death_on_impact]
                self.moving_tiles = False
                self.set_empty_positions()
                self.add_new_tile_not_random()

    def done_moving(self) -> bool:
        return all(not tile.death_on_impact for tile in self.tiles)

    def update(self) -> None:
        self.move()

    def set_tiles_from_history(self) -> None:
        self.tiles.clear()
        for i in range(2):
            pos = self.starting_positions[i, :2].astype(int)
            val = self.starting_positions[i, 2]
            tile = Tile(pos[0], pos[1], val)
            self.tiles.append(tile)
        self.remove_occupied_from_empty_positions()

    def remove_occupied_from_empty_positions(self) -> None:
        occupied_positions = [tile.position for tile in self.tiles]
        self.empty_positions = [
            pos
            for pos in self.empty_positions
            if not any(np.array_equal(pos, o_pos) for o_pos in occupied_positions)
        ]
