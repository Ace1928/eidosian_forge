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
class ThetaStar:

    def __init__(self, grid):
        """
        Initializes the ThetaStar object with the given grid object.
        This constructor sets up the necessary data structures and configurations to enable efficient
        pathfinding using the Theta* algorithm, tailored to the specific requirements of the game environment.

        Args:
            grid (Grid): The grid object representing the game environment.
        """
        self.grid = grid

    async def find_path(self, start, end):
        """
        Finds the shortest path from the start to the end point using the Theta* algorithm.
        This method employs an advanced variant of the A* algorithm, leveraging line-of-sight checks to
        optimize the path and reduce unnecessary node expansions, resulting in efficient and accurate pathfinding.

        Args:
            start (tuple): The starting point of the path.
            end (tuple): The endpoint of the path.

        Returns:
            list: The shortest path from start to end as a list of points, meticulously computed using the Theta*
            algorithm to ensure optimality and efficiency.
        """
        logging.debug(f'Initiating pathfinding from {start} to {end} using Theta*.')
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self.heuristic(start, end)
        while not open_set.empty():
            current = open_set.get()[1]
            if current == end:
                path = self.reconstruct_path(came_from, current)
                logging.info(f'Successfully computed shortest path from {start} to {end}: {' -> '.join(map(str, path))}')
                return path
            for neighbor in self.get_neighbors(current):
                if self.line_of_sight(came_from.get(current, current), neighbor):
                    tentative_g_score = g_score[current] + self.distance(came_from.get(current, current), neighbor)
                else:
                    tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    if neighbor not in open_set.queue:
                        open_set.put((f_score[neighbor], neighbor))
        logging.warning(f'Pathfinding from {start} to {end} using Theta* concluded with no path found.')
        return []

    def get_neighbors(self, position):
        """
        Retrieves the neighboring positions of a given position on the grid.
        This method considers the cardinal directions (up, down, left, right) as valid neighbors,
        ensuring a comprehensive exploration of the surrounding cells.

        Args:
            position (tuple): The position for which to retrieve the neighbors.

        Returns:
            list: A list of neighboring positions, each represented as a tuple of coordinates.
        """
        x, y = position
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [neighbor for neighbor in neighbors if 0 <= neighbor[0] < self.grid.width and 0 <= neighbor[1] < self.grid.height and (not self.grid.is_obstacle(neighbor))]

    def line_of_sight(self, start, end):
        """
        Determines whether there is a clear line of sight between two positions on the grid.
        This method uses the Bresenham's line algorithm to efficiently check for any obstacles or
        obstructions along the path, enabling the Theta* algorithm to make informed decisions about
        node expansion and path optimization.

        Args:
            start (tuple): The starting position of the line of sight check.
            end (tuple): The endpoint of the line of sight check.

        Returns:
            bool: True if there is a clear line of sight between the start and end positions, False otherwise.
        """
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while x1 != x2 or y1 != y2:
            if self.grid.is_obstacle((x1, y1)):
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return True

    def heuristic(self, position, goal):
        """
        Calculates the heuristic value (estimated cost) from a given position to the goal position.
        This method uses the Manhattan distance as the heuristic function, providing an admissible estimate
        of the remaining cost to reach the goal.

        Args:
            position (tuple): The position for which to calculate the heuristic value.
            goal (tuple): The goal position.

        Returns:
            float: The estimated cost from the position to the goal, calculated using the Manhattan distance.
        """
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def distance(self, start, end):
        """
        Calculates the Euclidean distance between two positions on the grid.
        This method provides an accurate measure of the cost of moving from one position to another,
        considering the diagonal distance when applicable.

        Args:
            start (tuple): The starting position.
            end (tuple): The endpoint.

        Returns:
            float: The Euclidean distance between the start and end positions.
        """
        return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

    def reconstruct_path(self, came_from, current):
        """
        Reconstructs the path from the start to the end position using the came_from map generated during the pathfinding process.
        This method traces back from the end position to the start, efficiently compiling the sequence of steps taken to reach the goal.

        Args:
            came_from (dict): A dictionary mapping each position to its predecessor along the path.
            current (tuple): The endpoint of the path.

        Returns:
            list: The reconstructed path as a list of positions, starting from the initial position and ending at the goal.
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path