import time
import pygame as pg
from pygame.math import Vector2
from typing import List, Set, Tuple, Optional
from random import randint
import numpy as np
import logging

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


class Node:
    def __init__(self, x: int, y: int):
        self.position: Vector2 = Vector2(x, y)
        self.parent: Optional["Node"] = None
        self.f: float = 0.0
        self.g: float = 0.0
        self.h: float = 0.0

    def is_equal(self, other: "Node") -> bool:
        return self.position == other.position


class SearchAlgorithm:
    def __init__(self, snake, apple):
        self.snake = snake
        self.apple = apple

    def get_path(self):
        start, end = self.get_start_end_positions()
        node_path = self.astar(start, end)
        vector_path = []
        for node in node_path:
            vector_path.append(Vector2(node.position.x, node.position.y))
        self.snake.path = vector_path

    def get_start_end_positions(self) -> Tuple[dict, dict]:
        start = {"x": int(self.snake.body[0].x), "y": int(self.snake.body[0].y)}
        end = {"x": int(self.apple.position[0]), "y": int(self.apple.position[1])}
        return start, end

    def astar(self, start, end):
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
            if current_node.is_equal(end_node):
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent
                possible_paths.append(list(reversed(path)))
            children = []
            for i in range(len(adjacent_squares)):
                node_position = [
                    int(current_node.position.x + adjacent_squares[i][0]),
                    int(current_node.position.y + adjacent_squares[i][1]),
                ]
                if 0 <= node_position[0] < self.snake.grid_size[0] // block_size:
                    if 0 <= node_position[1] < self.snake.grid_size[1] // block_size:
                        new_node = Node(node_position[0], node_position[1])
                        children.append(new_node)
            for i in range(len(children)):
                if_in_closed_list = False
                for j in range(len(closed_list)):
                    if children[i].is_equal(closed_list[j]):
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
                            children[i].is_equal(open_list[j])
                            and children[i].g < open_list[j].g
                        ):
                            present = True
                        elif (
                            children[i].is_equal(open_list[j])
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
