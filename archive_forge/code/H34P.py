import random
import numpy as np
import pygame
import sys
import pickle
from typing import List, Tuple, Optional, Callable
import pykan


class KAN:
    def __init__(
        self, input_size: int, output_size: int, depth: int, activation: str = "tanh"
    ):
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            depth (int): Number of layers in the network.
            activation (str): Activation function to use ('tanh', 'relu', 'sigmoid').
        """
        self.input_size = input_size
        self.output_size = output_size
        self.depth = depth
        self.activation = self._get_activation_function(activation)
        self.weights = [
            np.random.randn(input_size, output_size).astype(np.float32)
            for _ in range(depth)
        ]
        self.biases = [
            np.random.randn(output_size).astype(np.float32) for _ in range(depth)
        ]

    def _get_activation_function(self, name: str) -> Callable:
        """
        Get the activation function based on the name.

        Args:
            name (str): Name of the activation function.

        Returns:
            Callable: Activation function.
        """
        activations = {
            "tanh": np.tanh,
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        }
        if name in activations:
            return activations[name]
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after passing through the network.
        """
        for i in range(self.depth):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = self.activation(x)
        return x

    def mutate(self, mutation_rate: float = 0.1):
        """
        Mutate the network's weights and biases.

        Args:
            mutation_rate (float): Probability of mutation for each weight and bias.
        """
        for i in range(self.depth):
            if random.random() < mutation_rate:
                self.weights[i] += (
                    np.random.randn(*self.weights[i].shape).astype(np.float32)
                    * mutation_rate
                )
                self.biases[i] += (
                    np.random.randn(*self.biases[i].shape).astype(np.float32)
                    * mutation_rate
                )

    def inherit(self, other: "KAN"):
        """
        Inherit weights and biases from another KAN instance.

        Args:
            other (KAN): Another KAN instance to inherit from.
        """
        new_depth = max(
            1,
            self.depth
            + (1 if random.random() < 0.05 else -1 if random.random() < 0.05 else 0),
        )
        new_depth = min(new_depth, len(self.weights), len(other.weights))
        self.weights = [
            (self.weights[i] + other.weights[i]) / 2 for i in range(new_depth)
        ]
        self.biases = [(self.biases[i] + other.biases[i]) / 2 for i in range(new_depth)]
        self.depth = new_depth

    def save(self, filename: str):
        """
        Save the network's weights and biases to a file.

        Args:
            filename (str): The file path to save the network.
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {"weights": self.weights, "biases": self.biases, "depth": self.depth}, f
            )

    @classmethod
    def load(cls, filename: str) -> "KAN":
        """
        Load a network's weights and biases from a file.

        Args:
            filename (str): The file path to load the network from.

        Returns:
            KAN: An instance of the KAN class with loaded weights and biases.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            input_size=data["weights"][0].shape[0],
            output_size=data["biases"][0].shape[0],
            depth=data["depth"],
        )
        instance.weights = data["weights"]
        instance.biases = data["biases"]
        return instance

    def train(
        self,
        dataset: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        """
        Train the network using a simple gradient descent algorithm.

        Args:
            dataset (np.ndarray): Input data for training.
            targets (np.ndarray): Target outputs for training.
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            for x, y in zip(dataset, targets):
                output = self.forward(x)
                error = y - output
                self._backpropagate(x, error, learning_rate)

    def _backpropagate(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        """
        Backpropagate the error and update the weights and biases.

        Args:
            x (np.ndarray): Input data.
            error (np.ndarray): Error between predicted and actual output.
            learning_rate (float): Learning rate for gradient descent.
        """
        for i in reversed(range(self.depth)):
            delta = error * self._activation_derivative(self.activation, x)
            self.weights[i] += learning_rate * np.outer(x, delta)
            self.biases[i] += learning_rate * delta
            error = np.dot(delta, self.weights[i].T)

    def _activation_derivative(self, activation: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Args:
            activation (Callable): Activation function.
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Derivative of the activation function.
        """
        if activation == np.tanh:
            return 1 - np.tanh(x) ** 2
        elif activation == np.maximum:
            return np.where(x > 0, 1, 0)
        elif activation == (lambda x: 1 / (1 + np.exp(-x))):
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        else:
            raise ValueError(
                "Unsupported activation function for derivative calculation."
            )


# Game environment class
class GameEnvironment:
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
        self.resources = np.zeros(
            (size, size), dtype=[("quantity", np.float32), ("health", np.float32)]
        )
        self.populate_resources()

    def populate_resources(self):
        for _ in range(self.size):
            x, y = self.get_random_position()
            self.resources[x, y]["quantity"] += 1
            self.resources[x, y]["health"] = 1.0  # Initial health of the resource

    def place_creature(self, creature: "Creature", position: Tuple[int, int]):
        x, y = position
        self.grid[x, y] = creature
        creature.position = position

    def move_creature(
        self, old_position: Tuple[int, int], new_position: Tuple[int, int]
    ):
        x_old, y_old = old_position
        x_new, y_new = new_position
        self.grid[x_new, y_new] = self.grid[x_old, y_old]
        self.grid[x_old, y_old] = None
        if self.grid[x_new, y_new] is not None:
            self.grid[x_new, y_new].position = new_position

    def get_random_position(self) -> Tuple[int, int]:
        while True:
            position = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )
            if self.grid[position] is None:
                return position

    def respawn_resources(self):
        for _ in range(self.size // 10):  # Adjust respawn rate as needed
            x, y = self.get_random_position()
            self.resources[x, y]["quantity"] += 0.1
            self.resources[x, y][
                "health"
            ] = 1.0  # Reset health when new resource appears

    def decay_resources(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.resources[x, y]["quantity"] > 0:
                    self.resources[x, y]["health"] -= 0.01  # Resource health decay rate
                    if self.resources[x, y]["health"] <= 0:
                        self.resources[x, y][
                            "quantity"
                        ] = 0  # Remove resource if health is depleted

    def display(self):
        for row in self.grid:
            print(["." if cell is None else "C" for cell in row])
        print("\nResources:")
        for row in self.resources:
            print(row)
        print("\n")


# Creature class
class Creature:
    def __init__(
        self,
        health: float,
        attack: float,
        speed: float,
        intelligence: float,
        strategies: List[str],
        strategy_weights: np.ndarray,
        sequence_length: int,
        network: KAN,
        group_id: Optional[int] = None,
    ):
        self.health = health
        self.attack = attack
        self.speed = speed
        self.intelligence = intelligence
        self.strategies = strategies
        self.strategy_weights = strategy_weights
        self.sequence_length = sequence_length
        self.network = network
        self.movement_energy_cost = 0.1 * self.speed
        self.group_id = group_id
        self.position = None
        self.current_step = 0

    def battle(self, other: "Creature") -> bool:
        while self.health > 0 and other.health > 0:
            other.health -= self.attack
            if other.health <= 0:
                self.health += other.attack * 1.5  # Gain health from predation
                return True
            self.health -= other.attack
        return self.health > 0

    def mutate(self):
        mutation_factor = 0.1
        self.health = max(1.0, self.health + random.uniform(-1, 1) * mutation_factor)
        self.attack = max(1.0, self.attack + random.uniform(-1, 1) * mutation_factor)
        self.speed = max(1.0, self.speed + random.uniform(-1, 1) * mutation_factor)
        self.intelligence = max(
            1.0, self.intelligence + random.uniform(-1, 1) * mutation_factor
        )
        self.strategy_weights += (
            np.random.randn(*self.strategy_weights.shape).astype(np.float32)
            * mutation_factor
        )
        self.strategy_weights = np.clip(self.strategy_weights, 0, 1)
        self.network.mutate(mutation_rate=mutation_factor)
        self.sequence_length = max(
            1, int(self.sequence_length * (1 + random.uniform(-0.1, 0.1)))
        )

    def __repr__(self):
        return (
            f"Creature(Health: {self.health}, Attack: {self.attack}, Speed: {self.speed}, "
            f"Intelligence: {self.intelligence}, Strategies: {self.strategies}, "
            f"Weights: {self.strategy_weights}, Sequence: {self.sequence_length}, "
            f"Group: {self.group_id})"
        )

    def choose_move(
        self, current_position: Tuple[int, int], environment: "GameEnvironment"
    ) -> Tuple[int, int]:
        x, y = current_position
        best_move = (x, y)
        best_score = float("-inf")

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < environment.size and 0 <= ny < environment.size:
                if environment.grid[nx, ny] is None:
                    score = (
                        random.uniform(0, 1) * self.intelligence
                        + environment.resources[nx, ny]["quantity"]
                    )
                    if score > best_score:
                        best_move = (nx, ny)
                        best_score = score

        return best_move

    def acquire_resources(
        self, position: Tuple[int, int], environment: "GameEnvironment"
    ):
        x, y = position
        if environment.resources[x, y]["quantity"] > 0:
            self.health += 1
            environment.resources[x, y]["quantity"] = 0

    def lose_health(self):
        self.health -= 0.1  # Lose health over time if no resources are found

    def select_strategy(self) -> List[str]:
        return random.choices(
            self.strategies, self.strategy_weights, k=self.sequence_length
        )


# Battle class
class Battle:
    def __init__(self, environment: GameEnvironment):
        self.environment = environment

    def fight(self, position1: Tuple[int, int], position2: Tuple[int, int]):
        creature1 = self.environment.grid[position1]
        creature2 = self.environment.grid[position2]

        if creature1 and creature2:
            winner = creature1.battle(creature2)
            if winner:
                self.environment.grid[position2] = creature1
                creature1.position = position2
            else:
                self.environment.grid[position1] = creature2
                creature2.position = position1

    def handle_encounter(self, position1: Tuple[int, int], position2: Tuple[int, int]):
        creature1 = self.environment.grid[position1]
        creature2 = self.environment.grid[position2]

        if creature1 and creature2:
            strategies1 = creature1.select_strategy()
            strategies2 = creature2.select_strategy()
            for strategy1, strategy2 in zip(strategies1, strategies2):
                if strategy1 == "battle" or strategy2 == "battle":
                    self.fight(position1, position2)
                    break
                elif strategy1 == "flee" or strategy2 == "flee":
                    if strategy1 == "flee":
                        new_position = creature1.choose_move(
                            position1, self.environment
                        )
                        self.environment.move_creature(position1, new_position)
                        creature1.position = new_position
                    if strategy2 == "flee":
                        new_position = creature2.choose_move(
                            position2, self.environment
                        )
                        self.environment.move_creature(position2, new_position)
                        creature2.position = new_position
                    break
                elif strategy1 == "mate" and strategy2 == "mate":
                    self.mate(creature1, creature2, position1)
                    break
                elif strategy1 == "group" or strategy2 == "group":
                    self.group_defend(creature1, creature2, position1)
                    break

    def mate(self, creature1: Creature, creature2: Creature, position: Tuple[int, int]):
        new_health = (creature1.health + creature2.health) / 2.0
        new_attack = (creature1.attack + creature2.attack) / 2.0
        new_speed = (creature1.speed + creature2.speed) / 2.0
        new_intelligence = (creature1.intelligence + creature2.intelligence) / 2.0
        new_strategies = creature1.strategies
        new_weights = (creature1.strategy_weights + creature2.strategy_weights) / 2.0
        new_weights += (
            np.random.randn(*new_weights.shape).astype(np.float32) * 0.1
        )  # Mutate strategies
        new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1

        new_network = KAN(
            creature1.network.input_size,
            creature1.network.output_size,
            (creature1.network.depth + creature2.network.depth) // 2,
        )
        new_network.inherit(creature2.network)

        new_sequence_length = (
            creature1.sequence_length + creature2.sequence_length
        ) // 2
        if random.random() < 0.1:  # 10% chance to change sequence length
            new_sequence_length = max(1, new_sequence_length + random.choice([-1, 1]))

        new_creature = Creature(
            new_health,
            new_attack,
            new_speed,
            new_intelligence,
            new_strategies,
            new_weights,
            new_sequence_length,
            new_network,
        )
        self.environment.place_creature(
            new_creature, self.environment.get_random_position()
        )

    def group_defend(
        self, creature1: Creature, creature2: Creature, position: Tuple[int, int]
    ):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = position[0] + dx, position[1] + dy
            if 0 <= nx < self.environment.size and 0 <= ny < self.environment.size:
                neighbor = self.environment.grid[nx, ny]
                if neighbor and neighbor.group_id == creature1.group_id:
                    self.fight((nx, ny), position)


# Evolution class
class Evolution:
    def __init__(self, environment: GameEnvironment):
        self.environment = environment

    def evolve(self):
        new_creatures = []
        for x in range(self.environment.size):
            for y in range(self.environment.size):
                if self.environment.grid[x, y]:
                    creature = self.environment.grid[x, y]
                    creature.mutate()
                    new_creatures.append(creature)
        self.environment.grid = np.full(
            (self.environment.size, self.environment.size), None, dtype=object
        )
        for creature in new_creatures:
            self.environment.place_creature(
                creature, self.environment.get_random_position()
            )


# User interaction class
class UserInteraction:
    def __init__(
        self, environment: "GameEnvironment", evolution: Evolution, battle: Battle
    ):
        self.environment = environment
        self.evolution = evolution
        self.battle = battle

    def add_creature(
        self,
        health: float,
        attack: float,
        speed: float,
        intelligence: float,
        strategies: List[str],
        strategy_weights: np.ndarray,
        sequence_length: int,
        network: KAN,
        position: Tuple[int, int],
        group_id: Optional[int] = None,
    ):
        creature = Creature(
            health,
            attack,
            speed,
            intelligence,
            strategies,
            strategy_weights,
            sequence_length,
            network,
            group_id,
        )
        self.environment.place_creature(creature, position)

    def run_game(self, generations: int = 1000):
        pygame.init()
        screen_size = 1000
        cell_size = screen_size // self.environment.size
        screen = pygame.display.set_mode((screen_size, screen_size))
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            screen.fill((0, 0, 0))

            for generation in range(generations):
                self.evolution.evolve()
                self.environment.respawn_resources()
                self.environment.decay_resources()

                for x in range(self.environment.size):
                    for y in range(self.environment.size):
                        if self.environment.grid[x, y]:
                            creature = self.environment.grid[x, y]
                            new_position = creature.choose_move(
                                (x, y), self.environment
                            )
                            if new_position != (x, y):
                                self.environment.move_creature((x, y), new_position)
                                creature.lose_health()  # Lose health over time

                for x in range(self.environment.size):
                    for y in range(self.environment.size):
                        if self.environment.grid[x, y]:
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nx, ny = x + dx, y + dy
                                if (
                                    0 <= nx < self.environment.size
                                    and 0 <= ny < self.environment.size
                                ):
                                    if self.environment.grid[nx, ny]:
                                        self.battle.handle_encounter((x, y), (nx, ny))

                for x in range(self.environment.size):
                    for y in range(self.environment.size):
                        if self.environment.grid[x, y]:
                            pygame.draw.rect(
                                screen,
                                (0, 255, 0),
                                (x * cell_size, y * cell_size, cell_size, cell_size),
                            )
                        elif self.environment.resources[x, y]["quantity"] > 0:
                            pygame.draw.rect(
                                screen,
                                (0, 0, 255),
                                (x * cell_size, y * cell_size, cell_size, cell_size),
                            )

            pygame.display.flip()
            clock.tick(120)  # Adjust the speed as needed


# Function to save creatures to a file
def save_creatures(creatures: List[Creature], filename: str):
    with open(filename, "wb") as file:
        pickle.dump(creatures, file)


# Function to load creatures from a file
def load_creatures(filename: str) -> List[Creature]:
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    environment = GameEnvironment(size=100)
    evolution = Evolution(environment)
    battle = Battle(environment)
    interaction = UserInteraction(environment, evolution, battle)

    # Load creatures from file if available
    creatures = load_creatures("top_creatures.pkl")
    if creatures:
        for creature in creatures:
            interaction.add_creature(
                creature.health,
                creature.attack,
                creature.speed,
                creature.intelligence,
                creature.strategies,
                creature.strategy_weights,
                creature.sequence_length,
                creature.network,
                creature.position,
                creature.group_id,
            )
    else:
        # Define initial strategies and their weights
        strategies = ["battle", "flee", "group", "mate"]
        weights = np.array([0.2, 0.1, 0.3, 0.4], dtype=np.float32)

        # Create initial Kolmogorov-Arnold networks for creatures
        networks = [
            KAN(input_size=16, output_size=8, depth=i % 4 + 1) for i in range(12)
        ]

        # Add some initial creatures with group behavior
        for i, network in enumerate(networks):
            interaction.add_creature(
                health=random.uniform(8.0, 15.0),
                attack=random.uniform(2.0, 4.0),
                speed=random.uniform(1.0, 3.0),
                intelligence=random.uniform(4.0, 7.0),
                strategies=strategies,
                strategy_weights=weights,
                sequence_length=random.randint(3, 6),
                network=network,
                position=(i, i),
                group_id=i // 4 + 1,
            )

    # Run the game indefinitely
    interaction.run_game(generations=1)

    # Save top performing creatures
    top_creatures = sorted(
        environment.grid.flatten(), key=lambda c: c.health if c else 0, reverse=True
    )[:10]
    top_creatures = [creature for creature in top_creatures if creature]
    save_creatures(top_creatures, "top_creatures.pkl")
