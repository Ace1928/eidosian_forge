from pygame.math import Vector2
from Fruit import Fruit
from NN import NeuralNetwork
import pickle
import logging
import asyncio
import aiofiles
import logging
from typing import List, Optional, Tuple, Any, Set, Dict
import heapq
import threading
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utility import Grid, Node
from Algorithm import Algorithm
import numpy as np
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np
from NN import NeuralNetwork
from Utility import Node

# Configure logging to the highest level of verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
import numpy as np
from pygame.math import Vector2

from Algorithm import Algorithm
from Fruit import Fruit
from NN import NeuralNetwork
from Utility import Grid, Node

# Configure logging to the highest level of verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DecisionTree(Algorithm):
    """
    A decision tree algorithm for determining the best move for the snake based on the current game state and a trained neural network.
    """

    def __init__(
        self, snake: "Snake", network: NeuralNetwork, grid: List[List[Node]]
    ) -> None:
        """
        Initialize the DecisionTree object with the snake, neural network, and grid.

        Args:
            snake (Snake): The snake object for which decisions will be made.
            network (NeuralNetwork): The neural network that will be used for decision making.
            grid (List[List[Node]]): The grid representing the game environment.
        """
        super().__init__(grid)
        self.snake: "Snake" = snake
        self.network: NeuralNetwork = network
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=10)
        logging.debug(
            "DecisionTree object initialized with snake, neural network, and grid. Thread pool executor set up with 10 worker threads."
        )

    async def run_algorithm(self) -> Optional[Node]:
        """
        Execute the decision tree algorithm to determine the best move for the snake.

        Returns:
            Optional[Node]: The next node for the snake to move to, or None if no valid move is found.
        """
        logging.debug("Running decision tree algorithm to determine best move.")

        # Get current game state as input for neural network
        input_data: List[float] = await self.get_input_data()
        logging.debug(f"Input data for neural network: {input_data}")

        # Use neural network to predict best move
        output: np.ndarray = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.network.feed_forward,
            np.array(input_data, dtype=np.float32).reshape(1, -1),
        )
        logging.debug(f"Neural network output: {output}")

        # Interpret network output to determine best move
        best_move: Optional[Node] = await self.interpret_output(output)
        logging.debug(f"Best move determined: {best_move}")

        if best_move is None:
            logging.warning("No valid move found by decision tree.")

        return best_move

    async def get_input_data(self) -> List[float]:
        """
        Get the current game state as input data for the neural network.

        Returns:
            List[float]: The input data representing the current game state.
        """
        logging.debug("Getting current game state as input data for neural network.")

        # Get the snake's head position
        head_x, head_y = self.snake.head.x, self.snake.head.y

        # Get the fruit's position
        fruit_x, fruit_y = self.snake.fruit.x, self.snake.fruit.y

        # Calculate relative position of fruit to head
        fruit_relative_x = fruit_x - head_x
        fruit_relative_y = fruit_y - head_y

        # Get the direction of the snake
        direction = self.snake.direction

        # Check for obstacles around the head
        obstacle_left = await self.is_obstacle_at(head_x - 1, head_y)
        obstacle_right = await self.is_obstacle_at(head_x + 1, head_y)
        obstacle_up = await self.is_obstacle_at(head_x, head_y - 1)
        obstacle_down = await self.is_obstacle_at(head_x, head_y + 1)

        # Compile all data into a list
        input_data = [
            float(head_x),
            float(head_y),  # Snake head position
            float(fruit_relative_x),
            float(fruit_relative_y),  # Relative fruit position
            float(direction),  # Current direction of the snake
            float(obstacle_left),
            float(obstacle_right),
            float(obstacle_up),
            float(obstacle_down),  # Obstacles
        ]

        logging.debug(f"Input data collected: {input_data}")
        return input_data

    async def is_obstacle_at(self, x: int, y: int) -> bool:
        """
        Asynchronously check if there is an obstacle at the given position, ensuring thread safety and maximum concurrency.

        Args:
            x (int): The x-coordinate of the position to check.
            y (int): The y-coordinate of the position to check.

        Returns:
            bool: True if there is an obstacle at the given position, False otherwise.
        """
        logging.debug(f"Asynchronously checking for obstacle at position ({x}, {y})")

        # Check if the position is within the grid bounds
        if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
            logging.debug(f"Position ({x}, {y}) is outside the grid bounds.")
            return True

        # Asynchronously check if the position is occupied by the snake's body
        obstacle = await asyncio.gather(
            *[
                self.check_body_collision(x, y, body_part)
                for body_part in self.snake.body
            ]
        )

        if any(obstacle):
            logging.debug(f"Position ({x}, {y}) is occupied by the snake's body.")
            return True

        logging.debug(f"No obstacle found at position ({x}, {y}).")
        return False

    async def check_body_collision(self, x: int, y: int, body_part) -> bool:
        """
        Helper function to check collision with a body part asynchronously.

        Args:
            x (int): The x-coordinate of the position to check.
            y (int): The y-coordinate of the position to check.
            body_part (Node): A node representing a part of the snake's body.

        Returns:
            bool: True if the specified position collides with the body part, False otherwise.
        """
        return body_part.x == x and body_part.y == y

    async def interpret_output(self, output: np.ndarray) -> Optional[Node]:
        """
        Asynchronously interpret the neural network's output to determine the best move, ensuring thread safety and maximum concurrency.

        Args:
            output (np.ndarray): The output from the neural network.

        Returns:
            Optional[Node]: The node representing the best move, or None if no valid move.
        """
        logging.debug(f"Asynchronously interpreting neural network output: {output}")

        # Get the snake's current head position
        head_x, head_y = self.snake.head.x, self.snake.head.y

        # Define the possible moves based on the output
        moves = [
            (head_x, head_y - 1),  # Up
            (head_x + 1, head_y),  # Right
            (head_x, head_y + 1),  # Down
            (head_x - 1, head_y),  # Left
        ]

        # Find the index of the maximum value in the output
        max_index = int(np.argmax(output))

        # Asynchronously check if the move is valid (i.e., not an obstacle)
        if not await self.is_obstacle_at(moves[max_index][0], moves[max_index][1]):
            best_move = Node(moves[max_index][0], moves[max_index][1])
            logging.debug(f"Best move determined asynchronously: {best_move}")
            return best_move
        else:
            logging.warning(
                "Neural network output leads to an invalid move asynchronously."
            )
            return None

    async def train(self, reward: float) -> None:
        """
        Asynchronously train the neural network based on the reward from the last move, ensuring thread safety and maximum concurrency.

        Args:
            reward (float): The reward value from the last move.
        """
        logging.debug(f"Asynchronously training neural network with reward: {reward}")

        # Asynchronously get the input data and output from the last move
        input_data = await asyncio.to_thread(self.get_input_data)
        output = await asyncio.to_thread(
            self.network.feed_forward,
            np.array(input_data, dtype=np.float32).reshape(1, -1),
        )

        # Calculate the target output based on the reward asynchronously
        target_output = output.copy()
        target_output[0][np.argmax(output)] = reward

        # Asynchronously train the neural network using backpropagation
        await asyncio.to_thread(
            self.network.train,
            np.array(input_data, dtype=np.float32).reshape(1, -1),
            target_output,
        )

        logging.debug("Asynchronous neural network training complete.")

    async def save_model(self, file_path: str) -> None:
        """
        Asynchronously save the trained neural network model to a file, ensuring thread safety.

        Args:
            file_path (str): The path to save the model file to.
        """
        logging.debug(f"Asynchronously saving neural network model to: {file_path}")

        # Prepare the neural network weights and biases for saving
        data = {
            "weights": [w.tolist() for w in self.network.weights],
            "biases": [b.tolist() for b in self.network.biases],
        }

        # Asynchronously save the neural network weights and biases to a file
        async with aiofiles.open(file_path, "w") as file:
            await file.write(json.dumps(data, indent=4))

        logging.debug("Neural network model saved asynchronously and successfully.")

    async def load_model(self, file_path: str) -> None:
        """
        Asynchronously load a trained neural network model from a file, ensuring thread safety.

        Args:
            file_path (str): The path to load the model file from.
        """
        logging.debug(f"Asynchronously loading neural network model from: {file_path}")

        # Asynchronously load the neural network weights and biases from a file
        async with aiofiles.open(file_path, "r") as file:
            data = json.loads(await file.read())
        self.network.weights = [np.array(w, dtype=np.float32) for w in data["weights"]]
        self.network.biases = [np.array(b, dtype=np.float32) for b in data["biases"]]

        logging.debug("Neural network model loaded asynchronously and successfully.")


class Snake:
    """
    A class representing the Snake entity in a Snake game, equipped with AI capabilities.

    Attributes:
        body (List[Vector2]): A list of Vector2 objects representing the segments of the snake's body.
        fruit (Fruit): An instance of the Fruit class representing the fruit object in the game.
        score (int): The current score of the snake in the game.
        fitness (float): A metric representing the snake's performance.
        life_time (int): The total duration the snake has been alive in the game.
        steps (int): The number of steps the snake has taken.
        hidden (int): The number of hidden layers in the neural network.
        network (NeuralNetwork): An instance of the NeuralNetwork class used for AI decision making.
        learning_rate (float): The learning rate of the neural network.
        decision_tree (DecisionTree): An instance of the DecisionTree class used for AI decision making.
    """

    def __init__(self, hidden: int = 20, learning_rate: float = 0.1) -> None:
        """
        Initializes the Snake class with a specified number of hidden layers and learning rate for the neural network.

        Args:
            hidden (int): The number of hidden layers in the neural network. Default is 20.
            learning_rate (float): The learning rate of the neural network. Default is 0.1.
        """
        logging.debug(
            "Initializing the Snake class with meticulous detail. "
            f"Hidden layers: {hidden}, Learning rate: {learning_rate}"
        )
        self._body: List[Vector2] = [Vector2(5, 8), Vector2(4, 8), Vector2(3, 8)]
        self._fruit: Fruit = Fruit()
        self._score: int = 0
        self._fitness: float = 0.0
        self._life_time: int = 0
        self._steps: int = 0
        self._hidden: int = hidden
        self._learning_rate: float = learning_rate
        self._network: NeuralNetwork = NeuralNetwork(
            5, hidden, 3, learning_rate=learning_rate
        )
        self._decision_tree: Optional[DecisionTree] = None
        logging.debug(
            "Snake class initialized successfully with parameters: "
            f"body={self._body}, fruit={self._fruit}, score={self._score}, "
            f"fitness={self._fitness}, life_time={self._life_time}, steps={self._steps}, "
            f"hidden={self._hidden}, learning_rate={self._learning_rate}, network={self._network}"
        )

    async def save_model(self, network: NeuralNetwork, name: str) -> None:
        """
        Asynchronously saves the neural network model to a file.

        Args:
            network (NeuralNetwork): The neural network to be saved.
            name (str): The filename to save the model to.

        Raises:
            RuntimeError: If the model fails to save.
        """
        logging.debug(
            f"Attempting to save the model to {name} with detailed error handling."
        )
        try:
            async with aiofiles.open(name, "wb") as file:
                await file.write(pickle.dumps(network))
            logging.info(f"Model saved successfully in {name}.")
        except Exception as e:
            logging.error(f"Failed to save the model to {name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save the model due to: {e}") from e

    async def load_model(self, name: str) -> None:
        """
        Asynchronously loads the neural network model from a file.

        Args:
            name (str): The filename to load the model from.

        Raises:
            RuntimeError: If the model fails to load.
        """
        logging.debug(
            f"Attempting to load the model from {name} with detailed error handling."
        )
        try:
            async with aiofiles.open(name, "rb") as file:
                self._network = pickle.loads(await file.read())
            logging.info(f"Model loaded successfully from {name}.")
        except Exception as e:
            logging.error(f"Failed to load the model from {name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load the model due to: {e}") from e

    async def reset(self) -> None:
        """
        Resets the snake to its initial state.
        """
        logging.debug(
            "Resetting the Snake instance to its initial state with detailed logging."
        )
        self._body = [Vector2(5, 8), Vector2(4, 8), Vector2(3, 8)]
        await self._fruit.reset_seed()
        self._score = 0
        self._fitness = 0
        self._steps = 0
        self._network = NeuralNetwork(
            5, self._hidden, 3, learning_rate=self._learning_rate
        )
        logging.info(
            "Snake instance has been reset successfully with parameters reset to their initial values."
        )

    @property
    async def body(self) -> List[Vector2]:
        """
        Asynchronously retrieves the body of the snake ensuring thread safety.

        Returns:
            List[Vector2]: The body of the snake.
        """
        async with asyncio.Lock():
            return self._body.copy()

    @body.setter
    async def body(self, value: List[Vector2]) -> None:
        """
        Asynchronously sets the body of the snake ensuring thread safety.

        Args:
            value (List[Vector2]): The new body of the snake.
        """
        async with asyncio.Lock():
            self._body = value.copy()
            logging.debug(f"Body of the snake set asynchronously to {value}.")

    @property
    async def fruit(self) -> Fruit:
        """
        Asynchronously retrieves the fruit object in the game ensuring thread safety.

        Returns:
            Fruit: The fruit object.
        """
        async with asyncio.Lock():
            return self._fruit

    @fruit.setter
    async def fruit(self, value: Fruit) -> None:
        """
        Asynchronously sets the fruit object in the game ensuring thread safety.

        Args:
            value (Fruit): The new fruit object.
        """
        async with asyncio.Lock():
            self._fruit = value
            logging.debug(f"Fruit object set asynchronously to {value}.")

    @property
    async def score(self) -> int:
        """
        Asynchronously retrieves the score of the snake ensuring thread safety.

        Returns:
            int: The score of the snake.
        """
        async with asyncio.Lock():
            return self._score

    @score.setter
    async def score(self, value: int) -> None:
        """
        Asynchronously sets the score of the snake ensuring thread safety.

        Args:
            value (int): The new score of the snake.
        """
        async with asyncio.Lock():
            self._score = value
            logging.debug(f"Score of the snake set asynchronously to {value}.")

    @property
    async def fitness(self) -> float:
        """
        Asynchronously retrieves the fitness value of the snake ensuring thread safety.

        Returns:
            float: The fitness value of the snake.
        """
        async with asyncio.Lock():
            return self._fitness

    @fitness.setter
    async def fitness(self, value: float) -> None:
        """
        Asynchronously sets the fitness value of the snake ensuring thread safety.

        Args:
            value (float): The new fitness value of the snake.
        """
        async with asyncio.Lock():
            self._fitness = value
            logging.debug(f"Fitness value of the snake set asynchronously to {value}.")

    @property
    async def life_time(self) -> int:
        """
        Asynchronously retrieves the life time of the snake ensuring thread safety.

        Returns:
            int: The life time of the snake.
        """
        async with asyncio.Lock():
            return self._life_time

    @life_time.setter
    async def life_time(self, value: int) -> None:
        """
        Asynchronously sets the life time of the snake ensuring thread safety.

        Args:
            value (int): The new life time of the snake.

        Raises:
            ValueError: If the life time is negative.
        """
        if value < 0:
            raise ValueError("The life time of the snake cannot be negative.")
        async with asyncio.Lock():
            self._life_time = value
            logging.debug(f"Life time of the snake set asynchronously to {value}.")

    async def get_x(self) -> float:
        """
        Asynchronously retrieves the x-coordinate of the snake's head ensuring thread safety.

        Returns:
            float: The x-coordinate of the snake's head.
        """
        async with asyncio.Lock():
            logging.debug(
                "Asynchronously retrieving the x-coordinate of the snake's head."
            )
            return self.body[0].x

    async def get_y(self) -> float:
        """
        Asynchronously retrieves the y-coordinate of the snake's head ensuring thread safety.

        Returns:
            float: The y-coordinate of the snake's head.
        """
        async with asyncio.Lock():
            logging.debug(
                "Asynchronously retrieving the y-coordinate of the snake's head."
            )
            return self.body[0].y

    async def get_fruit(self) -> Vector2:
        """
        Asynchronously retrieves the position of the fruit ensuring thread safety.

        Returns:
            Vector2: The position of the fruit.
        """
        async with asyncio.Lock():
            logging.debug("Asynchronously retrieving the position of the fruit.")
            return self.fruit.position

    async def ate_fruit(self) -> bool:
        """
        Asynchronously checks if the snake has eaten the fruit ensuring thread safety.

        Returns:
            bool: True if the snake has eaten the fruit, False otherwise.
        """
        async with asyncio.Lock():
            logging.debug("Asynchronously checking if the snake has eaten the fruit.")
            if self.fruit.position == self.body[0]:
                self.score += 1
                self.life_time -= 40
                logging.info(
                    "Fruit has been eaten. Score incremented and life_time decremented."
                )
                return True
            return False

    async def create_fruit(self) -> None:
        """
        Asynchronously creates a new fruit in the game ensuring thread safety.
        """
        async with asyncio.Lock():
            logging.debug("Asynchronously creating a new fruit with detailed logging.")
            await self.fruit.generate_fruit()
            logging.info("New fruit created successfully.")

    async def move_ai(self, x: float, y: float) -> None:
        """
        Asynchronously moves the AI snake based on provided coordinates ensuring thread safety and maximum concurrency.

        Args:
            x (float): The x-coordinate to move to.
            y (float): The y-coordinate to move to.
        """
        async with asyncio.Lock():
            logging.debug(
                "Asynchronously moving AI based on provided coordinates with detailed logging."
            )
            self.life_time += 1
            self.steps += 1
            for i in range(len(self.body) - 1, 0, -1):
                self.body[i].x = self.body[i - 1].x
                self.body[i].y = self.body[i - 1].y

            self.body[0].x = x
            self.body[0].y = y
            logging.info("AI moved successfully with updated coordinates.")

    async def add_body_ai(self) -> None:
        """
        Asynchronously adds a new body segment to the AI snake ensuring thread safety and maximum concurrency.
        """
        async with asyncio.Lock():
            logging.debug(
                "Asynchronously adding a new body segment to the AI snake with detailed logging."
            )
            last_index: int = len(self.body) - 1
            tail: Vector2 = self.body[-1]
            before_last: Vector2 = self.body[-2]

            if tail.x == before_last.x:
                if tail.y < before_last.y:
                    self.body.append(Vector2(tail.x, tail.y - 1))
                else:
                    self.body.append(Vector2(tail.x, tail.y + 1))
            elif tail.y == before_last.y:
                if tail.x < before_last.x:
                    self.body.append(Vector2(tail.x - 1, tail.y))
                else:
                    self.body.append(Vector2(tail.x + 1, tail.y))
            logging.info(
                "New body segment added successfully with detailed position logging."
            )

    async def ate_body(self) -> bool:
        """
        Asynchronously checks if the snake has collided with its own body ensuring thread safety and maximum concurrency.

        Returns:
            bool: True if the snake has collided with its body, False otherwise.
        """
        async with asyncio.Lock():
            logging.debug(
                "Asynchronously checking if the snake has collided with its body with detailed logging."
            )
            for body_part in self.body[1:]:
                if self.body[0] == body_part:
                    logging.info("Collision detected: Snake has eaten its body.")
                    return True
            return False

    async def predict_move(self) -> int:
        """
        Asynchronously predicts the next move for the AI snake based on its vision, ensuring thread safety and maximum concurrency.

        Returns:
            int: The predicted move for the AI snake.
        """
        async with asyncio.Lock():
            logging.debug(
                "Asynchronously predicting the next move for the AI snake based on its vision."
            )
            if not hasattr(self, "decision_tree"):
                raise AttributeError(
                    "Decision tree model is not initialized in the AI snake."
                )
            predicted_move = await asyncio.to_thread(
                self.decision_tree.predict, self.vision
            )
            logging.info(f"Predicted move for the AI snake: {predicted_move}")
            return predicted_move

    async def predict_vision(self) -> List[float]:
        """
        Asynchronously predicts the vision for the AI snake, ensuring thread safety and maximum concurrency.

        Returns:
            List[float]: The predicted vision for the AI snake.
        """
        async with asyncio.Lock():
            logging.debug("Asynchronously predicting the vision for the AI snake.")
            if not hasattr(self, "decision_tree"):
                raise AttributeError(
                    "Decision tree model is not initialized in the AI snake."
                )
            predicted_vision = await asyncio.to_thread(
                self.decision_tree.predict, self.vision
            )
            logging.info(f"Predicted vision for the AI snake: {predicted_vision}")
            return predicted_vision
