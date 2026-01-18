def heuristic(
    self,
    a: Tuple[int, int],
    b: Tuple[int, int],
    last_dir: Optional[Tuple[int, int]] = None,
    is_return_path: bool = False,
) -> float:
    """
    Calculate the heuristic value for A* algorithm using the Euclidean distance.
    This heuristic is improved by adding a directional bias to discourage straight paths,
    promote zigzagging, and encourage dense packing. It also considers potential escape routes,
    avoids obstacles, boundaries, and the snake's body, and adjusts for the return path.

    Args:
        a (Tuple[int, int]): The current node coordinates.
        b (Tuple[int, int]): The goal node coordinates.
        last_dir (Optional[Tuple[int, int]]): The last direction moved.
        is_return_path (bool): Flag indicating if the heuristic is for the return path.

    Returns:
        float: The computed heuristic value.
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    manhattan_distance = dx + dy

    # Directional Bias: Penalize moving in the same direction to promote zigzagging
    direction_penalty = 0
    if last_dir:
        current_dir = (a[0] - b[0], a[1] - b[1])
        if current_dir == last_dir:
            direction_penalty = 5

    # Boundary Avoidance: Penalize nodes close to the grid boundaries
    boundary_threshold = 2
    boundary_penalty = 0
    if (
        a[0] < boundary_threshold
        or a[0] >= GRID_SIZE - boundary_threshold
        or a[1] < boundary_threshold
        or a[1] >= GRID_SIZE - boundary_threshold
    ):
        boundary_penalty = 10

    # Obstacle Avoidance: Penalize nodes that are adjacent to obstacles
    obstacle_penalty = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (a[0] + dx, a[1] + dy)
        if neighbor in self.obstacles:
            obstacle_penalty += 5

    # Snake Body Avoidance: Heavily penalize nodes that are part of the snake's body
    snake_body_penalty = 0
    if a in self.snake:
        snake_body_penalty = float("inf")

    # Escape Route: Favor nodes with more available neighboring nodes
    escape_route_bonus = 0
    available_neighbors = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (a[0] + dx, a[1] + dy)
        if (
            0 <= neighbor[0] < GRID_SIZE
            and 0 <= neighbor[1] < GRID_SIZE
            and neighbor not in self.snake
            and neighbor not in self.obstacles
        ):
            available_neighbors += 1
    escape_route_bonus = available_neighbors * -2

    # Dense Packing: Favor nodes that are closer to other parts of the snake's body
    dense_packing_bonus = 0
    for segment in self.snake:
        dense_packing_bonus += 1 / (
            1 + math.sqrt((a[0] - segment[0]) ** 2 + (a[1] - segment[1]) ** 2)
        )

    # Return Path: Adjust heuristic for the return path to prioritize reaching the tail
    return_path_bonus = 0
    if is_return_path:
        tail_distance = math.sqrt(
            (a[0] - self.snake[-1][0]) ** 2 + (a[1] - self.snake[-1][1]) ** 2
        )
        return_path_bonus = -tail_distance

    # Calculate the final heuristic value
    heuristic_value = (
        manhattan_distance
        + direction_penalty
        + boundary_penalty
        + obstacle_penalty
        + snake_body_penalty
        + escape_route_bonus
        + dense_packing_bonus
        + return_path_bonus
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

    return complete_path


def _a_star_search(
    self, start: Tuple[int, int], goal: Tuple[int, int], is_return_path: bool = False
) -> List[Tuple[int, int]]:
    """
    Helper function to perform the A* search algorithm for a single path.

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

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < GRID_SIZE
                and 0 <= neighbor[1] < GRID_SIZE
                and neighbor not in self.snake
                and neighbor not in self.obstacles
                and neighbor not in closed_set
            ):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal, (dx, dy), is_return_path
                    )
                    open_set.put((f_score[neighbor], neighbor))
                    last_direction = (dx, dy)

    return []
