"""
Module: search2048_overhauled.py
This module is part of the 2048 AI overhaul project, focusing on the game's search and decision-making mechanisms.
It includes the setup and main game loop, handling user input for tile movement, and comparing vector positions.

TODO:
- Integrate with AI decision-making components.
- Optimize performance for large search spaces.
- Expand functionality to support additional game features.
"""

# Importing necessary modules, classes, and libraries
from typing import Tuple, List, Optional  # Importing typing module for type hints
import numpy as np  # Importing numpy for array operations
from player_overhauled import (
    Player,
)  # Player class is defined in player_overhauled.py
import logging  # Importing logging module for debugging
import pygame  # Importing pygame for UI interactions
from standard_decorator import (
    StandardDecorator,
)  # Importing StandardDecorator class for function decorators
import unittest  # Importing unittest module for unit testing
import tracemalloc  # Importing tracemalloc for memory profiling

# Setting up logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initializing global variables
released: bool = True  # Flag to check if key is released
teleport: bool = False  # Flag to check if teleport is enabled

max_depth: int = 4  # Depth of search for AI decision making
pause_counter: int = 100  # Counter for pausing the game
next_connection_no: int = 1000  # Counter for next connection number
speed: int = 60  # Speed of the game
move_speed: int = 60  # Speed of tile movement

x_offset: int = 0  # X offset for tile movement
y_offset: int = 0  # Y offset for tile movement

# Placeholder for the Player instance
p: Optional[Player] = None  # Player instance


@StandardDecorator() # Applying standard decorator to setup function for retries, delay, and caching results for performance testing with default values.
def setup() -> None:
    """
    Sets up the game environment, including frame rate and window size, and initializes the player.
    Utilizes pygame for creating the game window and setting up the environment.
    """
    pygame.init()  # Initializing pygame
    screen = pygame.display.set_mode((850, 850))  # Setting window size
    pygame.display.set_caption("2048 AI Overhaul")  # Setting window title
    global p  # Accessing global Player instance
    p = Player()  # Initializing Player instance
    logging.debug("Game setup completed using pygame.")  # Logging setup completion


@Standar()      with default values.
def draw(screen: pygame.Surface) -> None:
    """
    Main game loop responsible for drawing the game state on each frame.
    It updates the background, draws the grid, and handles tile movements.
    Utilizes pygame for drawing the game state.

    Parameters:
        screen (pygame.Surface): The pygame screen surface to draw the game state.
    """
    screen.fill((187, 173, 160))  # Setting background color to light gray
    for i in range(4):  # Drawing grid lines using pygame lines for the 4x4 grid layout
        for j in range(
            4
        ):  # Drawing grid cells using pygame rectangles with appropriate colors
            pygame.draw.rect(  # Drawing grid cell using pygame
                screen,  # Drawing grid cell using pygame
                (205, 193, 180),  # Drawing grid cell using pygame
                (
                    i * 200 + (i + 1) * 10,
                    j * 200 + (j + 1) * 10,
                    200,
                    200,
                ),  # Drawing grid cell using pygame
            )
    pygame.display.update()

    if p and p.done_moving():  # Checking if player has finished moving tiles
        p.get_move()  # Getting the next move for the player
        p.move_tiles()  # Moving the tiles based on the player's decision


@Standar()      with default values.
def key_pressed(event: pygame.event.Event) -> None:
    """
    Handles key press events to control tile movements using pygame.

    Parameters:
        event (pygame.event.Event): The pygame event object containing key press information.
    """
    global released  # Accessing global key release flag
    if released:  # Checking if key is released
        if event.type == pygame.KEYDOWN:  # Checking if key is pressed
            if event.key == pygame.K_UP:  # Handling key press for moving tiles up
                if (
                    p and p.done_moving()
                ):  # Checking if player has finished moving tiles
                    p.move_direction = np.array([0, -1])  # Setting move direction to up
                    p.move_tiles()  # Moving the tiles in the specified direction
            elif event.key == pygame.K_DOWN:  # Handling key press for moving tiles down
                if (
                    p and p.done_moving()
                ):  # Checking if player has finished moving tiles
                    p.move_direction = np.array(
                        [0, 1]
                    )  # Setting move direction to down
                    p.move_tiles()  # Moving the tiles in the specified direction
            elif event.key == pygame.K_LEFT:  # Handling key press for moving tiles left
                if (
                    p and p.done_moving()
                ):  # Checking if player has finished moving tiles
                    p.move_direction = np.array(
                        [-1, 0]
                    )  # Setting move direction to left
                    p.move_tiles()  # Moving the tiles in the specified direction
            elif (
                event.key == pygame.K_RIGHT
            ):  # Handling key press for moving tiles right
                if (
                    p and p.done_moving()
                ):  # Checking if player has finished moving tiles
                    p.move_direction = np.array(
                        [1, 0]
                    )  # Setting move direction to right
                    p.move_tiles()  # Moving the tiles in the specified direction
        released = False  # Setting key release flag to False


@Standar()      with default values.
def key_released() -> None:
    """
    Resets the key release state to allow for new key press actions.
    This function is called when a key is released in the pygame event loop.
    """
    global released  # Accessing global key release flag
    released = True  # Setting key release flag to True


@Standar()      with default values.
def compare_vec(p1: np.array, p2: np.array) -> bool:
    """
    Compares two vectors for equality.

    Parameters:
        p1 (np.array): The first vector.
        p2 (np.array): The second vector.

    Returns:
        bool: True if the vectors are equal, False otherwise.
    """
    return np.array_equal(
        p1, p2
    )  # Comparing two vectors for equality so that the player can move tiles in the specified direction


class TestSearch2048Overhauled(unittest.TestCase):
    """
    This class contains unit tests for the search2048_overhauled module.
    It tests the functionality of setup, draw, key_pressed, key_released, and compare_vec functions.
    """

    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    def test_setup(self) -> None:
        """
        Tests the setup function to ensure the game environment is correctly initialized.
        """
        setup()  # Calling setup function
        self.assertIsNotNone(p, "Player instance should be initialized.")

    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    def test_draw(self) -> None:
        """
        Tests the draw function to ensure the game state is correctly drawn on the screen.
        """
        screen = pygame.display.set_mode((850, 850))  # Creating a test screen
        draw(screen)  # Calling draw function
        # Further tests could include checking the screen's state, but this requires a more complex setup.

    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    def test_key_pressed(self) -> None:
        """
        Tests the key_pressed function to ensure key press events are handled correctly.
        """
        # Simulating key press events would require integration testing with pygame event loop.

    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    def test_key_released(self) -> None:
        """
        Tests the key_released function to ensure the key release state is correctly reset.
        """
        key_released()  # Calling key_released function
        self.assertTrue(released, "Key release flag should be set to True.")

    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    def test_compare_vec(self) -> None:
        """
        Tests the compare_vec function to ensure vector comparison is accurate.
        """
        vec1 = np.array([1, 2])
        vec2 = np.array([1, 2])
        vec3 = np.array([2, 1])
        self.assertTrue(
            compare_vec(vec1, vec2), "Identical vectors should be considered equal."
        )
        self.assertFalse(
            compare_vec(vec1, vec3), "Different vectors should not be considered equal."
        )


# Running the unit tests if this module is run standalone
if __name__ == "__main__":
    tracemalloc.start()  # Starting memory profiling
    unittest.main()
