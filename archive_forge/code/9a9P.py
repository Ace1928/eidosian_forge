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


class Apple:
    """
    A class representing the apple object in the game, optimized with advanced data structures and algorithms.
    """

    def __init__(self, grid_size: Tuple[int, int] = (40, 20)) -> None:
        """
        Initialize the Apple object with a grid size, precomputing all possible positions for efficiency.

        Args:
            grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (40, 20).
        """
        self.grid_size: Tuple[int, int] = grid_size
        self.boxes: np.ndarray = np.array(
            [[i, j] for i in range(grid_size[0]) for j in range(grid_size[1])],
            dtype=np.int32,
        )
        self.position: np.ndarray = self.generate(
            np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int32)
        )

    def generate(self, snake_body: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a new position for the apple using optimized numpy operations for maximum efficiency.

        Args:
            snake_body (np.ndarray): The current positions of the snake body.

        Returns:
            Optional[np.ndarray]: The new position of the apple, or None if there are no available positions.
        """
        empty_boxes: np.ndarray = self.boxes[
            ~np.any(
                np.isin(self.boxes, snake_body).reshape(self.boxes.shape[0], -1), axis=1
            )
        ]

        if empty_boxes.size == 0:
            return None

        self.position = empty_boxes[np.random.choice(empty_boxes.shape[0])]
        return self.position

    def show(self, screen: pg.Surface) -> None:
        """
        Draw the apple on the screen using optimized Pygame methods for enhanced performance.

        Args:
            screen (pg.Surface): The surface to draw the apple on.
        """
        apple_rect: pg.Rect = pg.Rect(
            int(self.position[0] * 30), int(self.position[1] * 30), 30, 30
        )
        apple_surface: pg.Surface = pg.Surface((30, 30), flags=pg.SRCALPHA)
        apple_surface.fill((0, 0, 0, 0))
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-20, -20),
        )
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-10, -20),
        )
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-20, -10),
        )
        pg.draw.rect(
            apple_surface,
            pg.Color(255, 0, 0),
            apple_surface.get_rect().inflate(-10, -10),
        )
        screen.blit(apple_surface, apple_rect)


def main() -> None:
    """
    The main function to test and verify the functionality of the Apple class with comprehensive testing and validation.
    This function now includes rapid testing over 50 different positions and edge cases, with on-screen messaging to display test progress and results.
    """
    try:
        # Initialize Pygame
        pg.init()
        screen: pg.Surface = pg.display.set_mode((1200, 600))
        pg.display.set_caption("Apple Class Test")
        clock: pg.time.Clock = pg.time.Clock()
        font: pg.font.Font = pg.font.Font(None, 32)  # Default font, size 32

        # Create an instance of the Apple class
        apple: Apple = Apple()

        # Test the generate method over 50 different positions
        for i in range(50):
            snake_body: np.ndarray = np.array(
                [[randint(0, 29), randint(0, 19)] for _ in range(3)], dtype=np.int32
            )
            new_position: Optional[np.ndarray] = apple.generate(snake_body)
            assert (
                new_position is not None
            ), f"Failed to generate a new apple position at iteration {i}"
            assert not np.any(
                np.all(new_position == snake_body, axis=1)
            ), f"Generated apple position overlaps with snake body at iteration {i}"

            # Update apple's position for visual verification
            apple.position = new_position
            apple.draw(screen)  # Draw the apple at the new position

            message: str = (
                f"Test {i + 1}/50: New apple position generated successfully."
            )
            text_surface: pg.Surface = font.render(message, True, pg.Color("green"))
            screen.blit(text_surface, (50, 50 + i * 10))
            pg.display.flip()
            pg.time.delay(500)  # Pause to visually verify the apple's new position

            # Clear the screen after verification before the next position is generated
            screen.fill((0, 0, 0))  # Assuming black is the background color

        # Test the draw method
        apple.draw(screen)
        pg.display.flip()
        pg.time.delay(
            1000
        )  # Pause for 1 second to visually verify the apple is drawn correctly

        # Test edge case: generate apple when no empty positions are available
        grid_size: Tuple[int, int] = (3, 1)
        apple_edge_case: Apple = Apple(grid_size)
        snake_body_edge_case: np.ndarray = np.array(
            [[0, 0], [1, 0], [2, 0]], dtype=np.int32
        )
        new_position_edge_case: Optional[np.ndarray] = apple_edge_case.generate(
            snake_body_edge_case
        )
        assert (
            new_position_edge_case is None
        ), "Generated apple position when no empty positions available"

        message: str = "All tests passed successfully!"
        text_surface: pg.Surface = font.render(message, True, pg.Color("blue"))
        screen.blit(text_surface, (50, 550))
        pg.display.flip()
        pg.time.delay(2000)  # Pause to display the final message

    except AssertionError as e:
        message: str = f"Test failed: {str(e)}"
        text_surface: pg.Surface = font.render(message, True, pg.Color("red"))
        screen.blit(text_surface, (50, 550))
        pg.display.flip()
        pg.time.delay(2000)  # Pause to display the error message

    except Exception as e:
        message: str = f"An error occurred during testing: {str(e)}"
        text_surface: pg.Surface = font.render(message, True, pg.Color("red"))
        screen.blit(text_surface, (50, 550))
        pg.display.flip()
        pg.time.delay(2000)  # Pause to display the error message

    finally:
        pg.quit()


if __name__ == "__main__":
    main()
