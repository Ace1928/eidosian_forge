from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
import logging
import asyncio
from functools import lru_cache

# Setting up logging configuration with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Seed the random number generator for reproducibility, ensuring deterministic behavior
random.seed(USER_SEED)


class Population:
    population_start_size: int = 10000  # Initial size of the population
    hidden_node_count: int = (
        200  # Number of hidden nodes in each snake's neural network
    )

    def __init__(self) -> None:
        """
        Initialize the Population with empty lists of snakes and saved snakes.
        This constructor sets up the initial state of the Population object with no snakes.
        """
        self.snakes: list[Snake] = []  # Active snakes currently in the simulation
        self.saved_snakes: list[Snake] = (
            []
        )  # Snakes that have been removed from the simulation

        # Log the initialization of the Population
        logging.debug(
            "Population initialized with empty lists of snakes and saved snakes."
        )

    def initialize_population(self) -> None:
        """
        Populate the initial population of snakes, each with a unique neural network configuration.
        This method creates a new snake with a specified number of hidden nodes and adds it to the list of active snakes.
        """
        logging.debug("Initializing population of snakes with meticulous detail.")
        for _ in range(Population.population_start_size):
            new_snake = Snake(Population.hidden_node_count)
            self.snakes.append(new_snake)
            logging.debug(
                f"Added new snake with ID {id(new_snake)} to population, enhancing the genetic diversity."
            )

    def remove_snake(self, snake: Snake) -> None:
        """
        Remove a snake from the active population and add it to the saved snakes list.
        This method transitions a snake from being active in the population to being saved for future analysis or reproduction.

        Args:
            snake (Snake): The snake to be removed.
        """
        logging.debug(
            f"Removing snake with ID {id(snake)} from active population, transitioning to saved snakes."
        )
        self.saved_snakes.append(snake)
        self.snakes.remove(snake)
        logging.debug(
            f"Snake with ID {id(snake)} has been successfully moved to saved snakes, preserving its genetic data for future generations."
        )

        # Print statements for additional runtime information
        print(
            f"Snake with ID {id(snake)} removed from active population and added to saved snakes."
        )


class GA(Algorithm):
    """
    Genetic Algorithm (GA) class that extends the Algorithm base class to implement
    a genetic algorithm specifically tailored for optimizing the behavior of snakes
    in a grid-based simulation environment.

    Attributes:
        max_generations (int): Maximum number of generations the genetic algorithm will run.
        mutation_rate (float): Percentage of each snake's neural network that will be mutated.
        crossover_rate (float): Percentage of each parent's algorithm each child will inherit.
        elitism_rate (float): Percentage of the population that will pair up and create offspring each generation.
        offspring_per_generation (int): Number of new offspring made per pair of parents per generation.
        generations_until_death (int): Number of generations before an individual dies of old age.
        grid (np.ndarray): The grid on which the snakes will operate.
        population (Population): The population of snakes being evolved.
        current_generation (int): The current generation number in the evolutionary process.
        best_score (int): The highest score achieved by any snake in the population.
        best_generation (int): The generation number at which the best score was achieved.
        best_snake (Snake): The snake that achieved the best score.
    """

    max_generations: int = 10000
    mutation_rate: float = 0.20
    crossover_rate: float = 0.70
    elitism_rate: float = 0.10
    offspring_per_generation: int = 2
    generations_until_death: int = 10

    def __init__(self, grid: np.ndarray) -> None:
        """
        Initialize the genetic algorithm with a given grid and create the initial population.

        Args:
            grid (np.ndarray): The grid on which the snakes will operate.
        """
        super().__init__(grid)
        self.population = Population()
        self.current_generation: int = 0
        self.best_score: int = 0
        self.best_generation: int = 0
        self.best_snake: Optional[Snake] = None
        logging.debug("Genetic Algorithm initialized with grid and initial population.")

    def died(self, snake: Snake) -> None:
        """
        Process the death of a snake, checking if it collided with itself or boundaries.

        Args:
            snake (Snake): The snake to check for death conditions.
        """
        logging.debug(
            f"Processing potential death for snake with ID {id(snake)}, evaluating conditions."
        )
        current_x = snake.body[0].x
        current_y = snake.body[0].y

        if snake.ate_body() or snake.life_time > 80:
            logging.info(
                f"Snake with ID {id(snake)} died due to self-collision or exceeding life time threshold."
            )
            self.population.remove_snake(snake)

        elif (
            not 0 <= current_x < NO_OF_CELLS
            or not BANNER_HEIGHT <= current_y < NO_OF_CELLS
        ):
            logging.info(
                f"Snake with ID {id(snake)} died due to boundary collision, initiating removal."
            )
            self.population.remove_snake(snake)

    def generate_next_generation(self) -> bool:
        """
        Generate the next generation of snakes based on the current population's fitness.

        Returns:
            bool: False if the maximum generation limit is reached, True otherwise.
        """
        if self.current_generation == GA.max_generations:
            logging.info(
                "Maximum generation limit reached, halting further generation production."
            )
            return False

        self.calculate_fitness()
        self.identify_best_snake()
        self.perform_natural_selection()
        self.population.saved_snakes.clear()
        self.current_generation += 1
        logging.debug(
            f"Advanced to generation {self.current_generation}, preparing for new evolutionary challenges."
        )
        return True

    def is_population_extinct(self) -> bool:
        """
        Check if the current population of snakes is extinct.

        Returns:
            bool: True if there are no snakes left, False otherwise.
        """
        return len(self.population.snakes) == 0

    def identify_best_snake(self) -> Snake:
        """
        Identify the best performing snake from the saved snakes based on fitness.

        Returns:
            Snake: The snake with the highest fitness.
        """
        best_snake = max(self.population.saved_snakes, key=lambda snake: snake.fitness)
        logging.debug(
            f"Identified best snake with ID {id(best_snake)} and fitness {best_snake.fitness}, potentially setting a new benchmark for performance."
        )

        if best_snake.score > self.best_score:
            self.best_score = best_snake.score
            self.best_generation = self.current_generation
            self.best_snake = best_snake
            logging.info(
                f"New best snake found with score {self.best_score} at generation {self.best_generation}, marking a significant milestone in the evolutionary process."
            )

        return best_snake

    def check_directions(
        self, snake: Snake, direction_node: Node, inputs: list
    ) -> None:
        """
        Check the potential directions a snake can move to avoid collisions.

        This method evaluates possible movement directions for a given snake and appends a binary value to a list indicating whether a collision would occur if the snake moved in that direction. The method checks both boundary conditions and collisions with the snake's own body.

        Args:
            snake (Snake): The snake to check directions for.
            direction_node (Node): The node representing the direction to check.
            inputs (list): The list to append the result to, where 1 indicates a collision and 0 indicates no collision.

        Returns:
            None: This method does not return any value but modifies the 'inputs' list in place.
        """
        # Check if the direction node is outside the boundary or inside the snake's body
        if self.outside_boundary(direction_node) or self.inside_body(
            snake, direction_node
        ):
            inputs.append(1)  # Append 1 to inputs if there's a collision
        else:
            inputs.append(0)  # Append 0 to inputs if there's no collision

        # Log the result of the direction check for debugging purposes
        logging.debug(
            f"Direction check for snake ID {id(snake)} at node {direction_node}: {'Collision' if inputs[-1] == 1 else 'No Collision'}"
        )

    def run_algorithm(self, snake: Snake) -> tuple:
        """
        Run the genetic algorithm for a given snake, determining its next move based on neural network outputs and proximity to the fruit.

        This method calculates the potential movement directions for the snake, evaluates the distances to the fruit from these directions, and uses the snake's neural network to decide the optimal move. The method returns the coordinates of the chosen move.

        Args:
            snake (Snake): The snake to run the algorithm for.

        Returns:
            tuple: The coordinates (x, y) of the next move.
        """
        inputs: list = []  # Initialize the list to store inputs for the neural network
        fruit_node: Node = Node(
            snake.get_fruit().x, snake.get_fruit().y
        )  # Get the fruit's position as a node

        # Determine the head's current direction
        x: int = snake.body[0].x
        y: int = snake.body[0].y

        # Initialize nodes for potential movement directions
        forward: Node
        left: Node
        right: Node

        # Calculate forward, left, and right nodes based on the snake's current orientation
        if snake.body[1].x == x:
            if snake.body[1].y < y:
                # Snake is moving down
                forward = Node(x, y + 1)
                left = Node(x - 1, y)
                right = Node(x + 1, y)
            else:
                # Snake is moving up
                forward = Node(x, y - 1)
                left = Node(x + 1, y)
                right = Node(x - 1, y)
        elif snake.body[1].y == y:
            if snake.body[1].x < x:
                # Snake is moving right
                forward = Node(x + 1, y)
                left = Node(x, y - 1)
                right = Node(x, y + 1)
            else:
                # Snake is moving left
                forward = Node(x - 1, y)
                left = Node(x, y + 1)
                right = Node(x, y - 1)

        # Check potential directions for collisions
        self.check_directions(snake, forward, inputs)
        self.check_directions(snake, left, inputs)
        self.check_directions(snake, right, inputs)

        # Calculate Euclidean distances to the fruit from each direction
        forward_distance: float = self.euclidean_distance(fruit_node, forward)
        left_distance: float = self.euclidean_distance(fruit_node, left)
        right_distance: float = self.euclidean_distance(fruit_node, right)

        # Store distances in a list and find the index of the minimum distance
        distances: list[float] = [forward_distance, left_distance, right_distance]
        min_index: int = distances.index(min(distances))

        # Append the direction with the minimum distance to the inputs
        inputs.append(min_index)

        # Calculate the angle between the head and the fruit using vector operations
        head_vector: np.ndarray = np.array([int(snake.body[0].x), int(snake.body[0].y)])
        fruit_vector: np.ndarray = np.array([fruit_node.x, fruit_node.y])

        inner_product: float = np.inner(head_vector, fruit_vector)
        norms_product: float = np.linalg.norm(head_vector) * np.linalg.norm(
            fruit_vector
        )

        cosine_angle: float = round(inner_product / norms_product, 5)
        sine_angle: float = math.sqrt(1 - cosine_angle**2)
        inputs.append(sine_angle)

        # Feed inputs through the snake's neural network to get the outputs
        outputs: list = snake.network.feedforward(inputs)

        # Determine the best direction based on the highest output value
        max_index: int = outputs.tolist().index(max(outputs))
        chosen_direction: Node = {0: forward, 1: left, 2: right}[max_index]

        # Log the chosen direction for debugging purposes
        logging.debug(
            f"Snake with ID {id(snake)} will move to {chosen_direction.x}, {chosen_direction.y}, following the optimal path as determined by neural computation."
        )
        return chosen_direction.x, chosen_direction.y

    def select_parent(self) -> Snake:
        """
        Select a parent snake for reproduction based on its fitness.

        This method selects a parent snake from the saved snakes list using a fitness-proportionate selection algorithm, also known as roulette wheel selection. The method returns the selected parent snake.

        Returns:
            Snake: The selected parent snake.
        """
        index: int = 0
        r: float = random.random()  # Generate a random number between 0 and 1

        # Iterate through the saved snakes and subtract their fitness from 'r' until 'r' is less than 0
        while r > 0:
            r -= self.population.saved_snakes[index].fitness
            index += 1
        index -= 1  # Adjust index to get the correct snake

        selected_parent: Snake = self.population.saved_snakes[
            index
        ]  # Select the parent snake

        # Log the selection of the parent snake for debugging purposes
        logging.debug(
            f"Selected parent snake with ID {id(selected_parent)} for reproduction, ensuring genetic diversity and robustness in the next generation."
        )
        return selected_parent

    def perform_natural_selection(self) -> None:
        """
        Perform natural selection to create a new generation of snakes from the current population.

        This method simulates natural selection by selecting parent snakes based on their fitness, creating offspring through genetic crossover, and introducing mutations. The new generation of snakes replaces the current population.

        """
        new_snakes: list[Snake] = (
            []
        )  # Initialize a list to store the new generation of snakes

        # Create new snakes equal to the population size
        for _ in range(Population.population_size):
            parent_a: Snake = self.select_parent()  # Select the first parent
            parent_b: Snake = self.select_parent()  # Select the second parent
            child: Snake = Snake(
                Population.hidden_node_count
            )  # Create a new snake for the child

            # Perform genetic crossover and mutation
            child.network.crossover(parent_a.network, parent_b.network)
            child.network.mutate(GA.mutation_rate)

            new_snakes.append(child)  # Add the new snake to the list

            # Log the creation of the new snake for debugging purposes
            logging.debug(
                f"Created new snake with ID {id(child)} from parents {id(parent_a)} and {id(parent_b)}, contributing to the evolutionary progress through genetic crossover and mutation."
            )

        self.population.snakes = (
            new_snakes.copy()
        )  # Replace the current population with the new generation

        # Log the completion of natural selection for debugging purposes
        logging.info(
            "Completed natural selection and created new generation of snakes, ensuring the continuation and enhancement of the species through meticulous genetic management."
        )

    def calculate_fitness(self) -> None:
        """
        Calculate the fitness of each snake in the saved population based on their performance.

        This method calculates the fitness of each snake using a formula that considers the number of steps taken and the score achieved. The fitness values are then normalized to ensure fair comparison and selection.

        """
        for snake in self.population.saved_snakes:
            # Calculate fitness using a complex formula
            fitness: float = (snake.steps**3) * (3 ** (snake.score * 3)) - 1.5 ** (
                0.25 * snake.steps
            )
            snake.fitness = round(fitness, 7)  # Round the fitness value for precision

            # Log the calculated fitness for debugging purposes
            logging.debug(
                f"Calculated fitness for snake with ID {id(snake)}: {snake.fitness}, quantifying its performance and potential for survival in a competitive environment."
            )
        self.normalize_fitness_values()  # Normalize the fitness values

    def normalize_fitness_values(self) -> None:
        """
        Normalize the fitness values of the saved snakes to ensure fair comparison and selection.

        This method normalizes the fitness values of the saved snakes by dividing each fitness value by the total fitness of all saved snakes. This ensures that the selection process is based on relative performance.

        """
        total_fitness: float = sum(
            snake.fitness for snake in self.population.saved_snakes
        )  # Calculate the total fitness of all saved snakes

        for snake in self.population.saved_snakes:
            snake.fitness /= total_fitness  # Normalize the fitness value

            # Log the normalized fitness for debugging purposes
            logging.debug(
                f"Normalized fitness for snake with ID {id(snake)} to {snake.fitness}, ensuring equitable comparison and selection based on relative performance."
            )

    def done(self) -> bool:
        """
        Determine if the genetic algorithm process is complete by evaluating if the current generation has reached or exceeded the maximum allowable generations.

        This method provides a boolean output that can be utilized by external programs or modules to ascertain the completion status of the genetic algorithm's evolutionary process. It is essential for coordinating sequential or dependent operations that should only commence post the termination of this genetic algorithm.

        Returns:
            bool: True if the genetic algorithm has completed its process by reaching the maximum generations, False otherwise.
        """
        logging.debug(
            f"Initiating check to determine if the genetic algorithm has concluded. Current generation: {self.current_generation}, Maximum generations allowed: {GA.max_generations}."
        )
        completion_status: bool = self.current_generation >= GA.max_generations
        logging.debug(
            f"Completion status determined: {'completed' if completion_status else 'not completed'}."
        )
        return completion_status
