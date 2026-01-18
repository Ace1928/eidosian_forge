import numpy as np
import math
import logging
from Algorithm import Algorithm
from Constants import USER_SEED

# Setting up logging configuration with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Seed the numpy random number generator for reproducibility
np.random.seed(USER_SEED)


def sigmoid(m: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function."""
    logging.debug(f"Applying sigmoid activation function to the input: {m}")
    return 1 / (1 + np.exp(-m))


def ReLU(x: np.ndarray) -> np.ndarray:
    """Apply the Rectified Linear Unit (ReLU) activation function."""
    logging.debug(f"Applying ReLU activation function to the input: {x}")
    return x * (x > 0)


def tanh(x: np.ndarray) -> np.ndarray:
    """Apply the hyperbolic tangent activation function."""
    logging.debug(f"Applying tanh activation function to the input: {x}")
    return np.tanh(x)


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.shape = (input_nodes, hidden_nodes, output_nodes)
        logging.info(f"Initialized NeuralNetwork with shape {self.shape}")
        self.initialize()

    def initialize(self):
        """Initialize the weights and biases of the network."""
        self.biases = [
            np.random.randn(i) for i in [self.hidden_nodes, self.output_nodes]
        ]
        self.weights = [
            np.random.randn(j, i)
            for i, j in zip(
                [self.input_nodes, self.hidden_nodes],
                [self.hidden_nodes, self.output_nodes],
            )
        ]
        logging.debug(f"Network weights initialized: {self.weights}")
        logging.debug(f"Network biases initialized: {self.biases}")

    def feedforward(self, input_matrix: np.ndarray) -> np.ndarray:
        """Perform a feedforward pass through the network."""
        input_matrix = np.array(input_matrix)
        logging.debug(f"Starting feedforward pass with input: {input_matrix}")
        try:
            for b, w in zip(self.biases, self.weights):
                input_matrix = tanh(np.dot(w, input_matrix) + b)
                logging.debug(f"Feedforward pass, layer output: {input_matrix}")
        except Exception as e:
            logging.error(
                f"An error occurred during the feedforward pass: {e}", exc_info=True
            )
            raise RuntimeError(f"Feedforward pass failed due to: {e}") from e
        return input_matrix

    def crossover(self, networkA: "NeuralNetwork", networkB: "NeuralNetwork"):
        """Perform crossover between two networks."""
        weightsA = networkA.weights.copy()
        weightsB = networkB.weights.copy()

        biasesA = networkA.biases.copy()
        biasesB = networkB.biases.copy()

        for i in range(len(self.weights)):
            length = len(self.weights[i])
            split = np.random.randint(1, length)
            self.weights[i] = weightsA[i].copy()
            self.weights[i][split:] = weightsB[i][split:].copy()

        for i in range(len(self.biases)):
            length = len(self.biases[i])
            split = np.random.randint(1, length)
            self.biases[i] = biasesA[i].copy()
            self.biases[i][:split] = biasesB[i][:split].copy()
        logging.info("Crossover operation completed.")

    def mutation(self, a: float, val: float) -> float:
        """Mutate a weight or bias with a certain probability."""
        if np.random.rand() < val:
            mutated_value = np.random.randn()
            logging.debug(f"Mutating value {a} to {mutated_value}")
            return mutated_value
        return a

    def mutate(self, val: float):
        """Apply mutation to all weights and biases in the network."""
        mutation_function = np.vectorize(self.mutation)
        for i in range(len(self.weights)):
            self.weights[i] = mutation_function(self.weights[i], val)
        for i in range(len(self.biases)):
            self.biases[i] = mutation_function(self.biases[i], val)
        logging.info("Mutation operation completed.")

    def print(self):
        """Print the current state of the network."""
        print("shape", self.shape)
        print("weights", self.weights)
        print("biases", self.biases)
