import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Dict, Deque, Set, Optional
from heapq import heappush, heappop
import numpy as np
import math
from collections import PriorityQueue, defaultdict
import logging


# Setup logging configuration
logging.basicConfig(
    filename="snake_game_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Seed the random generator for reproducible results during development
seed(0)

# Initialize pygame
pg.init()
SCREEN_WIDTH: int = 1000
SCREEN_HEIGHT: int = 600
BLOCK_SIZE: int = 2
OPEN_SET_LIMIT: int = 1000
BEZIER_RESOLUTION: int = 10
window: pg.Surface = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Enhanced Snake Game with A* Pathfinding")


def heuristic(
    self,
    a: Tuple[int, int],
    b: Tuple[int, int],
    last_dir: Optional[Tuple[int, int]] = None,
    is_return_path: bool = False,
) -> float:
    """
    Calculate the heuristic value for A* algorithm using a dynamic, adaptive approach.
    This heuristic is optimized for real-time performance and scalability, incorporating
    multiple factors such as directional bias, obstacle avoidance, boundary awareness,
    snake body avoidance, escape route availability, dense packing, and path-specific
    adjustments. The heuristic is designed to generate strategic, efficient paths that
    adapt to the current game state and snake's length.

    Args:
        a (Tuple[int, int]): The current node coordinates.
        b (Tuple[int, int]): The goal node coordinates.
        last_dir (Optional[Tuple[int, int]]): The last direction moved.
        is_return_path (bool): Flag indicating if the heuristic is for the return path.

    Returns:
        float: The computed heuristic value.
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    euclidean_distance = math.sqrt(dx**2 + dy**2)

    # Directional Bias: Penalize moving in the same direction to promote zigzagging
    direction_penalty = 0
    if last_dir:
        current_dir = (a[0] - b[0], a[1] - b[1])
        if current_dir == last_dir:
            direction_penalty = 5 * (1 - len(self.snake) / (BLOCK_SIZE**2))

    # Boundary Awareness: Dynamically adjust penalty based on snake's proximity to boundaries
    boundary_threshold = max(2, int(0.1 * BLOCK_SIZE))
    boundary_penalty = 0
    if (
        a[0] < boundary_threshold
        or a[0] >= BLOCK_SIZE - boundary_threshold
        or a[1] < boundary_threshold
        or a[1] >= BLOCK_SIZE - boundary_threshold
    ):
        boundary_penalty = 10 * (1 - len(self.snake) / (BLOCK_SIZE**2))
        boundary_penalty *= (
            1
            - min(a[0], a[1], BLOCK_SIZE - a[0] - 1, BLOCK_SIZE - a[1] - 1)
            / boundary_threshold
        )

    # Obstacle Avoidance: Penalize nodes that are adjacent to obstacles, considering snake's length
    obstacle_penalty = 0
    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        neighbor = (a[0] + dx, a[1] + dy)
        if neighbor in self.obstacles:
            obstacle_penalty += 5 * (1 - len(self.snake) / (BLOCK_SIZE**2))

    # Snake Body Avoidance: Heavily penalize nodes that are part of the snake's body
    snake_body_penalty = 0
    if a in self.snake:
        snake_body_penalty = float("inf")

    # Escape Route: Favor nodes with more available neighboring nodes, considering snake's length
    escape_route_bonus = 0
    available_neighbors = 0
    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        neighbor = (a[0] + dx, a[1] + dy)
        if (
            0 <= neighbor[0] < BLOCK_SIZE
            and 0 <= neighbor[1] < BLOCK_SIZE
            and neighbor not in self.snake
            and neighbor not in self.obstacles
        ):
            available_neighbors += 1
    escape_route_bonus = available_neighbors * -2 * (len(self.snake) / (BLOCK_SIZE**2))

    # Dense Packing: Favor nodes that are closer to other parts of the snake's body, considering snake's length
    dense_packing_bonus = 0
    for segment in self.snake:
        dense_packing_bonus += 1 / (
            1 + math.sqrt((a[0] - segment[0]) ** 2 + (a[1] - segment[1]) ** 2)
        )
    dense_packing_bonus *= len(self.snake) / (BLOCK_SIZE**2)

    # Return Path: Dynamically adjust heuristic for the return path to prioritize reaching the tail
    return_path_bonus = 0
    if is_return_path:
        tail_distance = math.sqrt(
            (a[0] - self.snake[-1][0]) ** 2 + (a[1] - self.snake[-1][1]) ** 2
        )
        return_path_bonus = -tail_distance * (1 - len(self.snake) / (BLOCK_SIZE**2))

    # Food Seeking: Favor nodes that are closer to the food, considering snake's length
    food_seeking_bonus = 0
    if not is_return_path:
        food_distance = math.sqrt(
            (a[0] - self.food[0]) ** 2 + (a[1] - self.food[1]) ** 2
        )
        food_seeking_bonus = -food_distance * (1 - len(self.snake) / (BLOCK_SIZE**2))

    # Real-time Adaptation: Dynamically adjust heuristic weights based on game state
    snake_length_ratio = len(self.snake) / (BLOCK_SIZE**2)
    direction_penalty_weight = 1 - snake_length_ratio
    boundary_penalty_weight = 1 - snake_length_ratio
    obstacle_penalty_weight = 1 - snake_length_ratio
    escape_route_bonus_weight = snake_length_ratio
    dense_packing_bonus_weight = snake_length_ratio
    return_path_bonus_weight = snake_length_ratio
    food_seeking_bonus_weight = 1 - snake_length_ratio

    # Collision Avoidance: Penalize nodes that lead to potential collisions
    collision_penalty = 0
    if self._is_collision(a):
        collision_penalty = float("inf")

    # Adaptive Exploration: Encourage exploration of unexplored areas based on snake's length
    exploration_bonus = 0
    if not self._is_explored(a):
        exploration_bonus = 10 * (1 - len(self.snake) / (BLOCK_SIZE**2))

    # Calculate the final heuristic value
    heuristic_value = (
        euclidean_distance
        + direction_penalty * direction_penalty_weight
        + boundary_penalty * boundary_penalty_weight
        + obstacle_penalty * obstacle_penalty_weight
        + snake_body_penalty
        + escape_route_bonus * escape_route_bonus_weight
        + dense_packing_bonus * dense_packing_bonus_weight
        + return_path_bonus * return_path_bonus_weight
        + food_seeking_bonus * food_seeking_bonus_weight
        + collision_penalty
        + exploration_bonus
    )

    return heuristic_value


def a_star_search(
    self, start: Tuple[int, int], goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Perform the A* search algorithm to find the optimal path from start to goal and back to the tail.
    This implementation uses a priority queue to efficiently explore nodes, a closed set
    to avoid redundant processing, and a custom heuristic function that considers multiple
    factors to generate strategic and efficient paths. It also calculates a complete cycle by
    finding the path to the goal and then the path from the goal back to the snake's tail.
    The search is optimized for real-time performance, scalability, and adaptability to the
    current game state.

    Args:
        start (Tuple[int, int]): The starting position of the path.
        goal (Tuple[int, int]): The goal position of the path.

    Returns:
        List[Tuple[int, int]]: The optimal path from start to goal and back to the tail as a list of coordinates.
    """
    # Find the path from start to goal
    path_to_goal = self._a_star_search(start, goal, is_return_path=False)

    if not path_to_goal:
        return []

    # Find the path from goal to the snake's tail
    path_to_tail = self._a_star_search(
        path_to_goal[-1], self.snake[-1], is_return_path=True
    )

    # Combine the paths to create a complete cycle
    complete_path = path_to_goal + path_to_tail

    # Post-process the path to remove unnecessary zigzags and optimize for smoothness
    optimized_path = self._optimize_path(complete_path)

    # Check for collisions in the optimized path and adjust if necessary
    collision_free_path = self._avoid_collisions(optimized_path)

    return collision_free_path


def _a_star_search(
    self, start: Tuple[int, int], goal: Tuple[int, int], is_return_path: bool = False
) -> List[Tuple[int, int]]:
    """
    Helper function to perform the A* search algorithm for a single path.
    This implementation is optimized for real-time performance and scalability, using
    efficient data structures and pruning techniques to minimize the search space.
    It also incorporates adaptive heuristics and dynamic adjustments based on the current
    game state to generate strategic and efficient paths.

    Args:
        start (Tuple[int, int]): The starting position of the path.
        goal (Tuple[int, int]): The goal position of the path.
        is_return_path (bool): Flag indicating if the search is for the return path.

    Returns:
        List[Tuple[int, int]]: The optimal path from start to goal as a list of coordinates.
    """
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    g_score = defaultdict(lambda: float("inf"))
    g_score[start] = 0
    f_score = defaultdict(lambda: float("inf"))
    f_score[start] = self.heuristic(start, goal, is_return_path=is_return_path)

    closed_set = set()
    last_direction = None

    while not open_set.empty():
        current = open_set.get()[1]
        closed_set.add(current)

        if current == goal:
            return self.reconstruct_path(came_from, current)

        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < BLOCK_SIZE
                and 0 <= neighbor[1] < BLOCK_SIZE
                and neighbor not in self.snake
                and neighbor not in self.obstacles
                and neighbor not in closed_set
            ):
                tentative_g_score = g_score[current] + math.sqrt(dx**2 + dy**2)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal, (dx, dy), is_return_path
                    )
                    open_set.put((f_score[neighbor], neighbor))
                    last_direction = (dx, dy)

        # Prune the open set to remove low-priority nodes and maintain a manageable search space
        if open_set.qsize() > OPEN_SET_LIMIT:
            pruned_open_set = PriorityQueue()
            for _ in range(OPEN_SET_LIMIT):
                pruned_open_set.put(open_set.get())
            open_set = pruned_open_set

    return []


def _optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Optimize the path by removing unnecessary zigzags and smoothing the trajectory.
    This function applies a combination of techniques, including removing redundant nodes,
    smoothing sharp turns, and applying Bezier curve interpolation to generate a more
    natural and efficient path.

    Args:
        path (List[Tuple[int, int]]): The original path.

    Returns:
        List[Tuple[int, int]]: The optimized path.
    """
    if len(path) <= 2:
        return path

    # Remove redundant nodes
    optimized_path = [path[0]]
    for i in range(1, len(path) - 1):
        prev_direction = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_direction = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_direction != next_direction:
            optimized_path.append(path[i])
    optimized_path.append(path[-1])

    # Smooth sharp turns
    smoothed_path = [optimized_path[0]]
    for i in range(1, len(optimized_path) - 1):
        prev_node = optimized_path[i - 1]
        current_node = optimized_path[i]
        next_node = optimized_path[i + 1]
        if (
            current_node[0] != prev_node[0]
            and current_node[1] != prev_node[1]
            and current_node[0] != next_node[0]
            and current_node[1] != next_node[1]
        ):
            smoothed_path.append(
                (
                    (current_node[0] + prev_node[0]) // 2,
                    (current_node[1] + prev_node[1]) // 2,
                )
            )
            smoothed_path.append(
                (
                    (current_node[0] + next_node[0]) // 2,
                    (current_node[1] + next_node[1]) // 2,
                )
            )
        else:
            smoothed_path.append(current_node)
    smoothed_path.append(optimized_path[-1])

    # Apply Bezier curve interpolation for a smoother trajectory
    bezier_path = []
    for i in range(len(smoothed_path) - 1):
        p0 = smoothed_path[i]
        p1 = smoothed_path[i + 1]
        for t in range(BEZIER_RESOLUTION):
            t = t / BEZIER_RESOLUTION
            x = int((1 - t) * p0[0] + t * p1[0])
            y = int((1 - t) * p0[1] + t * p1[1])
            bezier_path.append((x, y))
    bezier_path.append(smoothed_path[-1])

    return bezier_path


def _avoid_collisions(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Check for collisions in the path and dynamically adjust the path to avoid them in real-time.
    This function iterates through the path and checks each node for potential collisions
    with obstacles, boundaries, or the snake's body. If a collision is detected, it
    employs an adaptive A* search to find an optimal alternative path around the collision point,
    considering the current game state and snake's trajectory. The collision avoidance is performed
    iteratively to ensure a collision-free path while maintaining path efficiency and smoothness.

    Args:
        path (List[Tuple[int, int]]): The original path.

    Returns:
        List[Tuple[int, int]]: The collision-free path.
    """
    collision_free_path = path.copy()
    path_length = len(collision_free_path)

    for i in range(path_length):
        node = collision_free_path[i]

        # Check for collisions with obstacles, boundaries, and snake's body
        if (
            node in self.obstacles
            or node[0] < 0
            or node[0] >= BLOCK_SIZE
            or node[1] < 0
            or node[1] >= BLOCK_SIZE
            or node in self.snake
        ):
            # Perform adaptive A* search to find an alternative path segment
            start = collision_free_path[max(0, i - 1)]
            goal = collision_free_path[min(i + 1, path_length - 1)]

            # Adjust heuristic based on the current game state and snake's trajectory
            adaptive_heuristic = lambda a, b: self.heuristic(
                a, b, self.direction, is_return_path=False
            )

            alternative_path = self._a_star_search(
                start, goal, adaptive_heuristic, is_return_path=False
            )

            # Replace the colliding segment with the alternative path
            if alternative_path:
                collision_free_path[max(0, i - 1) : min(i + 2, path_length)] = (
                    alternative_path
                )
            else:
                # If no alternative path found, remove the colliding node
                collision_free_path.pop(i)
                path_length -= 1
                i -= 1

    # Perform a final optimization pass to smooth the path
    optimized_collision_free_path = self._optimize_path(collision_free_path)

    return optimized_collision_free_path


def get_neighbors(node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return the neighbors of the given node within game boundaries."""
    directions = [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]
    return [
        (node[0] + dx, node[1] + dy)
        for dx, dy in directions
        if 0 <= node[0] + dx < SCREEN_WIDTH and 0 <= node[1] + dy < SCREEN_HEIGHT
    ]


class Fruit:
    def __init__(self):
        """
        Initializes a Fruit object with a position on the game window.
        """
        self.position: Tuple[int, int] = (0, 0)
        self.relocate()
        logging.info(f"Fruit placed at {self.position}")

    def draw(self) -> None:
        """
        Draw the fruit on the game window using a fixed color (red) and block size.
        """
        pg.draw.rect(window, (255, 0, 0), (*self.position, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Fruit drawn at {self.position}")

    def relocate(self, exclude=None) -> None:
        """
        Relocate the fruit to a random position within the game boundaries that is not occupied.
        Ensures the fruit does not spawn inside the snake's body.
        """
        if exclude is None:
            exclude = []
        while True:
            new_x: int = randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_y: int = randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position: Tuple[int, int] = (new_x, new_y)
            if new_position not in exclude:
                self.position = new_position
                break
        logging.info(f"Fruit relocated to {self.position}")


class Snake:
    def __init__(self):
        """
        Initializes the Snake object with a starting position and a fruit object.
        """
        self.body: Deque[Tuple[int, int]] = deque([(160, 160), (140, 160), (120, 160)])
        self.direction: Tuple[int, int] = (BLOCK_SIZE, 0)
        self.fruit = Fruit()
        self.path: List[Tuple[int, int]] = a_star(
            self.body[0], self.fruit.position, list(self.body)
        )
        logging.info("Snake initialized")

    def draw(self) -> None:
        """
        Draw the snake on the game window using a fixed color (green) and block size for each segment.
        """
        for segment in self.body:
            pg.draw.rect(window, (0, 255, 0), (*segment, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug("Snake drawn on window")

    def move(self) -> bool:
        """
        Move the snake based on the A* pathfinding result. Handles collision and game over scenarios.
        :return: Boolean indicating if the move was successful (True) or if the game is over (False).
        """
        if not self.path:
            self.path = a_star(self.body[0], self.fruit.position, list(self.body))
        if self.path:
            next_pos: Tuple[int, int] = self.path.pop(0)
            if (
                next_pos in self.body
                or not (0 <= next_pos[0] < SCREEN_WIDTH)
                or not (0 <= next_pos[1] < SCREEN_HEIGHT)
            ):
                logging.warning("Collision detected or snake out of bounds")
                return False
            self.body.appendleft(next_pos)
            if next_pos == self.fruit.position:
                self.fruit.relocate()
                self.path = a_star(self.body[0], self.fruit.position, list(self.body))
            else:
                self.body.pop()
            logging.info("Snake moved successfully")
            return True
        logging.warning("No path available, game over")
        return False


def main():
    clock = pg.time.Clock()
    snake = Snake()
    fruit = Fruit()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        window.fill((0, 0, 0))
        snake.fruit.draw()
        snake.draw()

        if not snake.move():
            logging.info("Game Over, restarting")
            print("Game Over")
            snake = Snake()  # Restart the game

        pg.display.flip()
        clock.tick(240)
        logging.debug("Game loop executed")


if __name__ == "__main__":
    main()
'''

import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Dict, Deque, Set

# Setup logging for debug and tracking
logging.basicConfig(level=logging.DEBUG, filename='snake_game.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize constants and pygame
SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE = 1000, 600, 20
pg.init()
window = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption('Advanced Snake Game with Dynamic Pathfinding')



class Fruit:
    def __init__(self):
        """ Initialize fruit at a random position. """
        self.position = (randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE,
                         randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE)

    def draw(self):
        """ Draw fruit on the window. """
        pg.draw.rect(window, (255, 0, 0), (*self.position, BLOCK_SIZE, BLOCK_SIZE))

    def relocate(self, snake_body: Deque[Tuple[int, int]]):
        """ Relocate fruit avoiding snake body. """
        while True:
            x, y = randint(0, SCREEN_WIDTH - BLOCK_SIZE, BLOCK_SIZE), randint(0, SCREEN_HEIGHT - BLOCK_SIZE, BLOCK_SIZE)
            if (x, y) not in snake_body:
                self.position = (x, y)
                break

class Snake:
    def __init__(self):
        """ Initialize snake with fixed starting position and direction. """
        self.body = deque([(400, 300), (380, 300), (360, 300)])
        self.direction = (BLOCK_SIZE, 0)
        self.fruit = Fruit()
        self.path = a_star(self.body[0], self.fruit.position, list(self.body))

    def draw(self):
        """ Draw the snake on the game window. """
        for segment in self.body:
            pg.draw.rect(window, (0, 255, 0), (*segment, BLOCK_SIZE, BLOCK_SIZE))

    def move(self):
        """ Move the snake based on pathfinding. """
        if not self.path:
            self.path = a_star(self.body[0], self.fruit.position, list(self.body))
        if self.path:
            next_position = self.path.pop(0)
            if next_position in self.body or 0 > next_position[0] >= SCREEN_WIDTH or 0 > next_position[1] >= SCREEN_HEIGHT:
                return False
            self.body.appendleft(next_position)
            if next_position == self.fruit.position:
                self.fruit.relocate(self.body)
            else:
                self.body.pop()
            return True
        return False

def main():
    """ Main function running the game loop. """
    clock = pg.time.Clock()
    snake = Snake()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        window.fill((0, 0, 0))
        snake.fruit.draw()
        snake.draw()

        if not snake.move():
            print("Game over. Restarting...")
            snake = Snake()

        pg.display.flip()
        clock.tick(10)

if __name__ == '__main__':
    main()
'''
