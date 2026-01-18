import unittest
import math
from A_STAR import A_STAR
from pygame.math import Vector2
from typing import List, Tuple, Any, Set
from Utility import Grid, Node
from Snake import Snake
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

grid = Grid(cells=20).get_grid()
grid_size = len(grid)


class TestA_STAR(unittest.TestCase):

    def setUp(self):
        self.grid = grid
        self.grid_size = len(self.grid)
        self.a_star = A_STAR(self.grid)

    def test_init(self):
        self.assertIsInstance(self.a_star, A_STAR)
        self.assertEqual(self.a_star.grid, self.grid)
        self.assertEqual(self.a_star.lock, threading.Lock())
        self.assertIsInstance(self.a_star.executor, ThreadPoolExecutor)
        self.assertEqual(self.a_star.executor._max_workers, 10)
        self.assertEqual(self.a_star.explored_set, set())
        self.assertEqual(self.a_star.frontier, [])

    def test_run_algorithm(self):
        snake = Snake()
        next_node = self.a_star.run_algorithm(snake)
        self.assertIsInstance(next_node, Node)

    def test_close(self):
        self.a_star.close()
        self.assertEqual(self.a_star.executor._shutdown, True)

    def test_get_neighbors_async(self):
        node = Node(1, 1)
        neighbors = self.a_star.get_neighbors_async(node)
        self.assertIsInstance(neighbors, list)
        self.assertEqual(await asyncio.run(len(neighbors)), 4)

    def test_find_neighbor(self):
        node = Node(1, 1)
        direction = Vector2(0, 1)
        neighbor = self.a_star.find_neighbor(node, direction)
        self.assertIsInstance(neighbor, Node)
        self.assertEqual(neighbor.x, 1)
        self.assertEqual(neighbor.y, 2)

    def test_valid_position(self):
        node = Node(1, 1)
        self.assertTrue(self.a_star.valid_position(node))

    def test_is_obstacle(self):
        node = Node(1, 1)
        self.assertFalse(self.a_star.is_obstacle(node))

    def test_process_neighbor(self):
        snake = Snake()
        current_node = Node(1, 1)
        neighbor = Node(1, 2)
        goalstate = Node(4, 4)
        self.a_star.process_neighbor(snake, current_node, neighbor, goalstate)
        self.assertEqual(neighbor.g, 2)
        self.assertEqual(neighbor.h, 6)
        self.assertEqual(neighbor.f, 8)

    def test_calculate_heuristic(self):
        goalstate = Node(4, 4)
        neighbor = Node(1, 1)
        heuristic = self.a_star.calculate_heuristic(goalstate, neighbor)
        self.assertEqual(heuristic, 3.5, 0.4)

    def test_select_optimal_path(self):
        path = [Node(1, 1), Node(1, 2), Node(1, 3), Node(1, 4)]
        optimal_path = self.a_star.select_optimal_path()
        self.assertEqual(optimal_path, path)

    def test_calculate_space_score(self):
        path = [Node(1, 1), Node(1, 2), Node(1, 3), Node(1, 4)]
        space_score = self.a_star.calculate_space_score(path)
        self.assertEqual(space_score, 3)

    def test_calculate_future_moves_score(self):
        path = [Node(1, 1), Node(1, 2), Node(1, 3), Node(1, 4)]
        future_moves_score = self.a_star.calculate_future_moves_score(path)
        self.assertEqual(future_moves_score, 3)


if __name__ == "__main__":
    unittest.main()
