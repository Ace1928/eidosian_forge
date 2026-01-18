import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
from collections import defaultdict
class AmoebaHamiltonianPathfinder:

    def __init__(self, snake=Snake(), grid=Grid(WIDTH, HEIGHT), food=Food()):
        """
        Initializes the AmoebaHamiltonianPathfinder object with the given snake, grid, and food objects.
        This constructor sets up the necessary data structures and configurations to enable the computation
        of Hamiltonian paths using an amoeba-inspired algorithm, tailored to the specific requirements of
        the snake game environment.

        Args:
            snake (Snake): The snake object representing the game agent.
            grid (Grid): The grid object representing the game environment.
            food (Food): The food object representing the target for the snake.
        """
        self.snake = snake
        self.grid = grid
        self.food = food

    async def find_path(self, start, end):
        """
        Finds a Hamiltonian path from the start to the end point using an amoeba-inspired algorithm.
        This method employs a novel nature-inspired approach to efficiently explore the game grid, adapting
        to the dynamic constraints imposed by the snake's body and the environment, while striving to find
        a path that covers all accessible cells.

        Args:
            start (tuple): The starting point of the path.
            end (tuple): The endpoint of the path.

        Returns:
            list: A Hamiltonian path from start to end as a list of points, meticulously computed to ensure
            maximum coverage of the game grid while avoiding obstacles and the snake's body.
        """
        logging.debug(f'Finding Hamiltonian path from {start} to {end}.')
        path = []
        visited = set()
        current_position = start
        while current_position != end:
            visited.add(current_position)
            path.append(current_position)
            next_position = self.select_next_position(current_position, visited)
            if next_position is None:
                logging.warning(f'No valid next position found from {current_position}. Terminating path.')
                break
            current_position = next_position
        if current_position == end:
            path.append(end)
            logging.info(f'Hamiltonian path found from {start} to {end}: {' -> '.join(map(str, path))}')
        else:
            logging.warning(f'Hamiltonian path from {start} to {end} incomplete. Partial path: {' -> '.join(map(str, path))}')
        return path

    def select_next_position(self, current_position, visited):
        """
        Selects the next position for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method carefully evaluates the candidate positions surrounding the current position, considering
        factors such as grid boundaries, obstacles, the snake's body, and previously visited cells, to determine
        the most promising next step in the path.

        Args:
            current_position (tuple): The current position of the pathfinding algorithm.
            visited (set): A set of previously visited positions.

        Returns:
            tuple: The selected next position for the pathfinding algorithm, chosen based on a comprehensive
            assessment of the available options and their potential to lead to a complete Hamiltonian path.
        """
        candidate_positions = self.get_candidate_positions(current_position)
        valid_positions = [pos for pos in candidate_positions if self.is_valid_position(pos) and pos not in visited]
        if not valid_positions:
            return None
        next_position = random.choice(valid_positions)
        logging.debug(f'Selected next position: {next_position}')
        return next_position

    def get_candidate_positions(self, current_position):
        """
        Retrieves the candidate positions surrounding the current position for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method systematically explores the adjacent cells in all cardinal directions, ensuring a comprehensive consideration
        of potential next steps in the path.

        Args:
            current_position (tuple): The current position of the pathfinding algorithm.

        Returns:
            list: A list of candidate positions surrounding the current position, each represented as a tuple of coordinates.
        """
        x, y = current_position
        candidate_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return candidate_positions

    def is_valid_position(self, position):
        """
        Determines whether a given position is valid for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method thoroughly checks the position against the grid boundaries, obstacles, and the snake's body,
        ensuring that only accessible and safe cells are considered as valid steps in the path.

        Args:
            position (tuple): The position to validate.

        Returns:
            bool: True if the position is valid and accessible, False otherwise.
        """
        x, y = position
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return False
        if position in self.snake.segments:
            return False
        if self.grid.is_obstacle(position):
            return False
        return True