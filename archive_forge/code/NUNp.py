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
import standard_decorator  # Importing standard_decorator module for standardizing function signatures

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


def key_released() -> None:
    """
    Resets the key release state to allow for new key press actions.
    This function is called when a key is released in the pygame event loop.
    """
    global released  # Accessing global key release flag
    released = True  # Setting key release flag to True


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
