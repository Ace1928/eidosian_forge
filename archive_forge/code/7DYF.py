from pygame.math import Vector2
from Fruit import Fruit
from NN import NeuralNetwork
import pickle
import logging

# Configure logging to the highest level of verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Snake:
    def __init__(self, hidden: int = 8) -> None:
        logging.debug("Initializing the Snake class with meticulous detail.")
        self.body: list[Vector2] = [Vector2(5, 8), Vector2(4, 8), Vector2(3, 8)]
        self.fruit: Fruit = Fruit()

        self.score: int = 0
        self.fitness: float = 0.0

        self.life_time: int = 0
        self.steps: int = 0
        self.hidden: int = hidden
        self.network: NeuralNetwork = NeuralNetwork(5, self.hidden, 3)
        logging.debug(
            "Snake class initialized successfully with parameters: body={}, fruit={}, score={}, fitness={}, life_time={}, steps={}, hidden={}, network={}".format(
                self.body,
                self.fruit,
                self.score,
                self.fitness,
                self.life_time,
                self.steps,
                self.hidden,
                self.network,
            )
        )

    def save_model(self, network: NeuralNetwork, name: str) -> None:
        logging.debug(
            f"Attempting to save the model to {name} with detailed error handling."
        )
        try:
            with open(name, "wb") as file:
                pickle.dump(network, file)
            logging.info(f"Model saved successfully in {name}.")
        except Exception as e:
            logging.error(f"Failed to save the model to {name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save the model due to: {e}") from e

    def load_model(self, name: str) -> None:
        logging.debug(
            f"Attempting to load the model from {name} with detailed error handling."
        )
        try:
            with open(name, "rb") as file:
                self.network = pickle.load(file)
            logging.info(f"Model loaded successfully from {name}.")
        except Exception as e:
            logging.error(f"Failed to load the model from {name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load the model due to: {e}") from e

    def reset(self) -> None:
        logging.debug(
            "Resetting the Snake instance to its initial state with detailed logging."
        )
        self.body = [Vector2(5, 8), Vector2(4, 8), Vector2(3, 8)]
        self.fruit.reset_seed()

        self.score = 0
        self.fitness = 0
        self.steps = 0

        self.network = NeuralNetwork(5, self.hidden, 3)
        logging.info(
            "Snake instance has been reset successfully with parameters reset to their initial values."
        )

    def get_x(self) -> float:
        logging.debug("Retrieving the x-coordinate of the snake's head.")
        return self.body[0].x

    def get_y(self) -> float:
        logging.debug("Retrieving the y-coordinate of the snake's head.")
        return self.body[0].y

    def get_fruit(self) -> Vector2:
        logging.debug("Retrieving the position of the fruit.")
        return self.fruit.position

    def ate_fruit(self) -> bool:
        logging.debug("Checking if the snake has eaten the fruit.")
        if self.fruit.position == self.body[0]:
            self.score += 1
            self.life_time -= 40
            logging.info(
                "Fruit has been eaten. Score incremented and life_time decremented."
            )
            return True
        return False

    def create_fruit(self) -> None:
        logging.debug("Creating a new fruit with detailed logging.")
        self.fruit.generate_fruit()
        logging.info("New fruit created successfully.")

    def move_ai(self, x: float, y: float) -> None:
        logging.debug("Moving AI based on provided coordinates with detailed logging.")
        self.life_time += 1
        self.steps += 1
        for i in range(len(self.body) - 1, 0, -1):
            self.body[i].x = self.body[i - 1].x
            self.body[i].y = self.body[i - 1].y

        self.body[0].x = x
        self.body[0].y = y
        logging.info("AI moved successfully with updated coordinates.")

    def add_body_ai(self) -> None:
        logging.debug(
            "Adding a new body segment to the AI snake with detailed logging."
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

    def ate_body(self) -> bool:
        logging.debug(
            "Checking if the snake has collided with its body with detailed logging."
        )
        for body_part in self.body[1:]:
            if self.body[0] == body_part:
                logging.info("Collision detected: Snake has eaten its body.")
                return True
        return False
