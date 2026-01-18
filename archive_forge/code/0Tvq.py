import pygame
import random
import logging
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Optional, Set

# Initialize logging
logging.basicConfig(
    filename="pygame_snake_game.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Game configuration
CELL_SIZE = 20
GRID_SIZE = 20
FPS = 60
GAME_TICK = 10


class SnakeGameAI:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        )
        pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self) -> None:
        self.snake: List[Tuple[int, int]] = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.score: int = 0
        self.food: Optional[Tuple[int, int]] = None
        self.game_over: bool = False
        self.direction: Tuple[int, int] = (1, 0)  # Start moving right
        self.path: List[Tuple[int, int]] = []
        self.tail_path: List[Tuple[int, int]] = []
        self.place_food()
        logging.info("Game reset.")

    def place_food(self) -> None:
        while self.food is None:
            potential_food = (
                random.randint(1, GRID_SIZE - 2),
                random.randint(1, GRID_SIZE - 2),
            )
            if potential_food not in self.snake:
                self.food = potential_food
                logging.debug(f"Food placed at {self.food}")

    def play_step(self) -> bool:
        if self.game_over:
            return self.game_over
        self.move_snake()
        self.check_collision()
        return self.game_over

    def move_snake(self) -> None:
        head_x, head_y = self.snake[0]
        new_x = head_x + self.direction[0]
        new_y = head_y + self.direction[1]
        new_head = (new_x, new_y)
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.place_food()
            self.path = []
            self.tail_path = []
        else:
            self.snake.pop()

    def check_collision(self) -> None:
        head = self.snake[0]
        if (
            head in self.snake[1:]
            or head[0] < 1
            or head[0] >= GRID_SIZE - 1
            or head[1] < 1
            or head[1] >= GRID_SIZE - 1
        ):
            self.game_over = True
            logging.error("Collision detected: Game over.")

    def draw_elements(self) -> None:
        self.screen.fill(BLACK)
        # Draw borders
        pygame.draw.rect(self.screen, WHITE, [0, 0, GRID_SIZE * CELL_SIZE, CELL_SIZE])
        pygame.draw.rect(
            self.screen,
            WHITE,
            [0, (GRID_SIZE - 1) * CELL_SIZE, GRID_SIZE * CELL_SIZE, CELL_SIZE],
        )
        pygame.draw.rect(self.screen, WHITE, [0, 0, CELL_SIZE, GRID_SIZE * CELL_SIZE])
        pygame.draw.rect(
            self.screen,
            WHITE,
            [(GRID_SIZE - 1) * CELL_SIZE, 0, CELL_SIZE, GRID_SIZE * CELL_SIZE],
        )
        # Draw path
        for x, y in self.path + self.tail_path:
            pygame.draw.rect(
                self.screen,
                YELLOW,
                [x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE],
                2,
            )
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = ((i * 50) % 255, (i * 100) % 255, (i * 150) % 255)
            pygame.draw.rect(
                self.screen, color, [x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE]
            )
        # Draw food
        if self.food:
            color = RED if pygame.time.get_ticks() % 500 < 250 else PINK
            pygame.draw.rect(
                self.screen,
                color,
                [
                    self.food[0] * CELL_SIZE,
                    self.food[1] * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ],
            )
        pygame.display.update()

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_search(
        self, start: Tuple[int, int], goal: Tuple[int, int], avoid: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heappush(open_set, (0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self.heuristic(start, goal)}

        while open_set:
            current = heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    1 <= neighbor[0] < GRID_SIZE - 1
                    and 1 <= neighbor[1] < GRID_SIZE - 1
                    and neighbor not in self.snake
                    and neighbor not in avoid
                ):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(
                            neighbor, goal
                        )
                        heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_next_direction(self) -> Tuple[int, int]:
        if not self.path:
            self.path = self.a_star_search(self.snake[0], self.food, set(self.snake))
            self.tail_path = self.a_star_search(
                self.food, self.snake[-1], set(self.snake + self.path)
            )

        if len(self.path) > 1:
            next_pos = self.path[1]
            self.path = self.path[1:]
            dx = next_pos[0] - self.snake[0][0]
            dy = next_pos[1] - self.snake[0][1]
            return dx, dy
        elif len(self.tail_path) > 1:
            next_pos = self.tail_path[1]
            self.tail_path = self.tail_path[1:]
            dx = next_pos[0] - self.snake[0][0]
            dy = next_pos[1] - self.snake[0][1]
            return dx, dy
        else:
            self.path = self.a_star_search(self.snake[0], self.food, set(self.snake))
            self.tail_path = self.a_star_search(
                self.food, self.snake[-1], set(self.snake + self.path)
            )
            if len(self.path) > 1:
                next_pos = self.path[1]
                self.path = self.path[1:]
                dx = next_pos[0] - self.snake[0][0]
                dy = next_pos[1] - self.snake[0][1]
                return dx, dy

        return self.direction

    def run(self) -> None:
        running = True
        game_tick = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if game_tick % (FPS // GAME_TICK) == 0:
                self.direction = self.get_next_direction()
                game_over = self.play_step()
                if game_over:
                    self.reset_game()

            self.draw_elements()
            self.clock.tick(FPS)
            game_tick += 1

        pygame.quit()


if __name__ == "__main__":
    game = SnakeGameAI()
    game.run()
