# // Creating the snake, apple and A* search algorithm
# Creating the snake, apple and search algorithm
import snake
import apple
import search

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


def calculate_block_size(screen_width: int, screen_height: int) -> int:
    """
    Calculate the block size based on screen resolution to ensure visibility and proportionality.
    Define a scaling function for block size relative to screen resolution.
    """
    reference_resolution = (1920, 1080)
    reference_block_size = 20

    scaling_factor_width = screen_width / reference_resolution[0]
    scaling_factor_height = screen_height / reference_resolution[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


BLOCK_SIZE = calculate_block_size(display_info.current_w, display_info.current_h)
border_width = 3 * BLOCK_SIZE
SCREEN_SIZE = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)
BORDER_COLOR = (255, 255, 255)
snake_instance = snake.Snake()
apple_instance = apple.Apple()
search_algorithm = search.Search(snake_instance, apple_instance)
game_clock = pg.time.Clock()
FRAMES_PER_SECOND = 60
TICK_RATE = 1000 // FRAMES_PER_SECOND


def setup() -> (
    Tuple[pg.Surface, snake.Snake, apple.Apple, search.Search, pg.time.Clock]
):
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    pg.init()
    screen: pg.Surface = pg.display.set_mode(SCREEN_SIZE)
    snake: snake.Snake = snake_instance
    apple: apple.Apple = apple_instance
    search: search.Search = search_algorithm
    search.get_path()
    clock: pg.time.Clock = game_clock
    return screen, snake, apple, search, clock


def draw(
    screen: pg.Surface, snake: snake.Snake, apple: apple.Apple, clock: pg.time.Clock
) -> None:
    """
    Manages the drawing of all game objects on the screen.
    This function separates game logic updates from rendering for modularity and flexibility.
    """
    screen.fill((51, 51, 51))
    pg.draw.rect(
        screen,
        BORDER_COLOR,
        pg.Rect(
            0, 0, SCREEN_SIZE[0] + 2 * border_width, SCREEN_SIZE[1] + 2 * border_width
        ),
        border_width,
    )
    snake.show(screen)
    apple.show(screen)
    snake.update(apple)
    pg.display.flip()
    clock.tick(FRAMES_PER_SECOND)
