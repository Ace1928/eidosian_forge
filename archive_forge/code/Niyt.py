# Creating the snake, apple and search algorithm
import snake
import apple
import search
import logging
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

# Configure logging
logging.basicConfig(
    filename="game.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def calculate_block_size(screen_width: int, screen_height: int) -> int:
    """
    Calculate the block size based on screen resolution to ensure visibility and proportionality.
    Define a scaling function for block size relative to screen resolution.
    """
    REFERENCE_RESOLUTION = (1920, 1080)
    REFERENCE_BLOCK_SIZE = 20

    scaling_factor_width = screen_width / REFERENCE_RESOLUTION[0]
    scaling_factor_height = screen_height / REFERENCE_RESOLUTION[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    dynamic_block_size = max(1, int(REFERENCE_BLOCK_SIZE * scaling_factor))
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)

    logging.info(f"Calculated block size: {adjusted_block_size}")
    return adjusted_block_size


block_size = calculate_block_size(display_info.current_w, display_info.current_h)
border_width = 3 * block_size
screen_size = (
    display_info.current_w - 2 * border_width,
    display_info.current_h - 2 * border_width,
)
border_color = (255, 255, 255)
snake_instance = snake.Snake()
apple_instance = apple.Apple()
search_algorithm = search.SearchAlgorithm(snake_instance, apple_instance)
game_clock = pg.time.Clock()
frames_per_second = 60
tick_rate = 1000 // frames_per_second


def setup_environment() -> (
    Tuple[pg.Surface, snake.Snake, apple.Apple, search.SearchAlgorithm, pg.time.Clock]
):
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    pg.init()
    screen: pg.Surface = pg.display.set_mode(screen_size)
    snake_entity: snake.Snake = snake_instance
    apple_entity: apple.Apple = apple_instance
    search_entity: search.SearchAlgorithm = search_algorithm
    search_entity.get_path()
    clock: pg.time.Clock = game_clock

    logging.info("Game environment setup complete")
    return screen, snake_entity, apple_entity, search_entity, clock


def draw_game_elements(
    screen: pg.Surface,
    snake_entity: snake.Snake,
    apple_entity: apple.Apple,
    clock: pg.time.Clock,
) -> None:
    """
    Manages the drawing of all game objects on the screen.
    This function separates game logic updates from rendering for modularity and flexibility.
    """
    screen.fill((51, 51, 51))
    pg.draw.rect(
        screen,
        border_color,
        pg.Rect(
            0, 0, screen_size[0] + 2 * border_width, screen_size[1] + 2 * border_width
        ),
        border_width,
    )
    snake_entity.show(screen)
    apple_entity.show(screen)
    snake_entity.update(apple_entity)
    pg.display.flip()
    clock.tick(frames_per_second)

    logging.debug("Game elements drawn on the screen")


def main() -> None:
    """
    The main function orchestrates the entire game by integrating components from snake.py, apple.py, and search.py.
    It initializes the game environment, processes user inputs, updates game states, and renders the game elements.
    """
    # Initialize the game environment
    screen, snake_entity, apple_entity, search_entity, game_clock = setup_environment()

    logging.info("Starting main game loop")

    # Main game loop
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                logging.info("Quit event received. Exiting game.")
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    snake_entity.change_direction(0, -1)
                elif event.key == pg.K_DOWN:
                    snake_entity.change_direction(0, 1)
                elif event.key == pg.K_LEFT:
                    snake_entity.change_direction(-1, 0)
                elif event.key == pg.K_RIGHT:
                    snake_entity.change_direction(1, 0)
                logging.debug(f"Key press event: {pg.key.name(event.key)}")

        # Update the game state
        snake_entity.update(apple_entity)
        if apple_entity.position in snake_entity.body:
            apple_entity.position = apple_entity.generate(snake_entity.body)
            logging.info("Snake collided with apple. Generating new apple position.")

        # Pathfinding update
        search_entity.get_path(apple_entity.position)
        logging.debug("Updated pathfinding")

        # Render the game elements
        draw_game_elements(screen, snake_entity, apple_entity, game_clock)

        # Refresh the screen
        pg.display.flip()
        game_clock.tick(frames_per_second)

    # Clean up and close the game properly
    pg.quit()
    logging.info("Game exited successfully")


if __name__ == "__main__":
    main()
