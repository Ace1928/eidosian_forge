# This is the A* pathfinding algorithm
# This works by finding the longest possible path between
# the snake's head and the snake's tail
# The snake will never get trapped because the snake's head
# will always have a way out after reaching the previous tail position
# Apple will be eaten when the snake is on the path

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
CLOCK = pg.time.Clock()
FPS = 60
TICK_RATE = 1000 // FPS


# Initial setup of the game environment
def setup() -> Tuple[pg.Surface, pg.time.Clock]:
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    # Initialize Pygame
    pg.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pg.Surface = pg.display.set_mode(SCREEN_SIZE)
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock: pg.time.Clock = CLOCK
    return screen, clock


class Node:
    def __init__(self, x: int, y: int):
        self.position: Vector2 = Vector2(x, y)
        self.parent: Optional["Node"] = None
        self.f: float = 0.0
        self.g: float = 0.0
        self.h: float = 0.0

    def equals(self, other: "Node") -> bool:
        return self.position == other.position


class Search:
    def __init__(self, snake, apple):
        self.snake = snake
        self.apple = apple

    def refreshMaze(self):
        maze = []
        for i in range(20):
            row = []
            for j in range(40):
                row.append(0)
            maze.append(row)
        for i in range(len(self.snake.body)):
            maze[self.snake.body[i].y][self.snake.body[i].x] = -1
        head_position = self.snake.getHeadPosition()
        tail_position = self.snake.getTailPosition()
        maze[head_position.y][head_position.x] = 1
        maze[tail_position.y][tail_position.x] = 2
        return maze

    def getPath(self):
        maze = self.refreshMaze()
        start, end = None, None
        for i in range(40):
            for j in range(20):
                if maze[j][i] == 1:
                    start = {"x": i, "y": j}
                elif maze[j][i] == 2:
                    end = {"x": i, "y": j}
        node_path = self.astar(maze, start, end)
        vector_path = []
        for i in range(len(node_path)):
            vector_path.append(Vector2(node_path[i]["x"], node_path[i]["y"]))
        self.snake.path = vector_path

    def astar(self, maze, start, end):
        start_node = Node(start["x"], start["y"])
        end_node = Node(end["x"], end["y"])
        open_list = []
        closed_list = []
        open_list.append(start_node)
        possible_paths = []
        adjacent_squares = [
            [0, -1],
            [0, 1],
            [-1, 0],
            [1, 0],
        ]
        while len(open_list) > 0:
            current_node = open_list[0]
            current_index = 0
            index = 0
            for i in range(len(open_list)):
                if open_list[i].f > current_node.f:
                    current_node = open_list[i]
                    current_index = index
                index += 1
            open_list.pop(current_index)
            closed_list.append(current_node)
            if current_node.equals(end_node):
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent
                possible_paths.append(list(reversed(path)))
            children = []
            for i in range(len(adjacent_squares)):
                node_position = [
                    current_node.position.x + adjacent_squares[i][0],
                    current_node.position.y + adjacent_squares[i][1],
                ]
                if 0 <= node_position[0] <= 39:
                    if 0 <= node_position[1] <= 19:
                        if maze[node_position[1]][node_position[0]] != -1:
                            new_node = Node(node_position[0], node_position[1])
                            children.append(new_node)
            for i in range(len(children)):
                if_in_closed_list = False
                for j in range(len(closed_list)):
                    if children[i].equals(closed_list[j]):
                        if_in_closed_list = True
                if not if_in_closed_list:
                    children[i].g = current_node.g + 2
                    children[i].h = abs(
                        children[i].position.x - end_node.position.x
                    ) + abs(children[i].position.y - end_node.position.y)
                    children[i].f = children[i].g + children[i].h
                    present = False
                    for j in range(len(open_list)):
                        if (
                            children[i].equals(open_list[j])
                            and children[i].g < open_list[j].g
                        ):
                            present = True
                        elif (
                            children[i].equals(open_list[j])
                            and children[i].g >= open_list[j].g
                        ):
                            open_list[j] = children[i]
                            open_list[j].parent = current_node
                    if not present:
                        children[i].parent = current_node
                        open_list.append(children[i])
        path = []
        for i in range(len(possible_paths)):
            if len(possible_paths[i]) > len(path):
                path = possible_paths[i]
        return path
