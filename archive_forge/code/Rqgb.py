# // Creating the snake, apple and A* search algorithm
# Creating the snake, apple and search algorithm
# let snake;
import snake

# let apple;
import apple

# let search;
import search

# Rest of the Imports required in alignment with the rest of the classes.
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint


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


# // Setting up everything
# Initial setup of the game environment
# function setup() {
def setup():
    # Initialize Pygame
    pg.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen = pg.display.set_mode(SCREEN_SIZE)
    # Instantiate the Snake object
    snake = snake.Snake()
    # Instantiate the Apple object
    apple = apple.Apple()
    # Instantiate the Search object with snake and apple as parameters
    search = search.Search(snake, apple)
    # Initiate the pathfinding algorithm
    search.get_path()
    # Set the frame rate
    clock = pg.time.Clock()
    return screen, snake, apple, search, clock


# This function manages the drawing of all game objects on the screen.
# function draw() {
def draw(screen, snake, apple, clock):
    # Set the background color
    screen.fill((51, 51, 51))
    # Draw the border around the screen using the BORDER_COLOR and border_width constants
    pg.draw.rect(
        screen,
        BORDER_COLOR,
        pg.Rect(
            0, 0, SCREEN_SIZE[0] + 2 * border_width, SCREEN_SIZE[1] + 2 * border_width
        ),
        border_width,
    )
    # Display the snake on the screen
    snake.show()
    # Display the apple on the screen
    apple.draw(screen)
    # Update the snake's position considering the apple
    snake.update(apple)
    # Update the display to reflect changes
    pg.display.flip()
    # Control the frame rate
    clock.tick(200)
