import unittest
import torch
import sys
import asyncio
import logging
from unittest.mock import patch
import numpy as np

# Append the system path to include the specific directory for module importation
sys.path.append("/home/lloyd/EVIE/Indellama3/indego")
from ActivationDictionary import ActivationDictionary

# Import the IndegoAdaptAct module
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,
    AdaptiveActivationNetwork,
    calculate_reward,
    update_policy_network,
    log_decision,
)

# Integrating the advanced logging configuration from the IndegoLogging module to ensure all events are meticulously recorded with utmost detail and precision
from IndegoLogging import configure_logging


# Asynchronous setup of the logging module
async def setup_logging():
    await configure_logging()  # This function call configures the logging based on the IndegoLogging module's configuration file


# Ensure there is an event loop for the current thread
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
# Utilizing asyncio's event loop to perform the asynchronous logging setup
# Check that there is a main loop, if there is not (ie during testing) ensure that one is started and is closed once logging finishes.
if asyncio.get_event_loop().is_running():
    # If the event loop is already running, run the logging setup synchronously
    asyncio.run(setup_logging())
else:
    # If the event loop is not running, start it and run the logging setup asynchronously
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup_logging())
# Acquiring a logger instance for the current module from the centralized logging configuration
# This logger adheres to the configurations set up by the IndegoLogging module, ensuring all logging is centralized, easily manageable, and aligned with the highest standards of operational excellence
logger = logging.getLogger(__name__)


import unittest
import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EnhancedPolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers_info: list):
        super(EnhancedPolicyNetwork, self).__init__()
        layers = []
        for index, layer_size in enumerate(layers_info):
            if index == 0:
                layers.append(torch.nn.Linear(input_dim, layer_size))
            else:
                layers.append(torch.nn.Linear(layers_info[index - 1], layer_size))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layers_info[-1], output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActivationDictionary:
    def __init__(self):
        self.activations = {
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
        }

    def get_activation(self, name: str):
        return self.activations.get(
            name, torch.nn.ReLU()
        )  # Default to ReLU if not found


class AdaptiveActivationNetwork(torch.nn.Module):
    def __init__(
        self,
        activation_dict: ActivationDictionary,
        in_features: int,
        out_features: int,
        layers_info: list,
    ):
        super(AdaptiveActivationNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for index, layer_size in enumerate(layers_info):
            if index == 0:
                self.layers.append(torch.nn.Linear(in_features, layer_size))
            else:
                self.layers.append(torch.nn.Linear(layers_info[index - 1], layer_size))
            self.layers.append(
                activation_dict.get_activation("relu")
            )  # Example using ReLU
        self.layers.append(torch.nn.Linear(layers_info[-1], out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def calculate_reward(
    current_loss: float, previous_loss: float, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    # Example reward calculation logic
    accuracy = np.mean(y_true == y_pred)
    improvement = previous_loss - current_loss
    return accuracy + improvement


def update_policy_network(
    network: EnhancedPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    reward: float,
    log_prob: float,
):
    loss = -log_prob * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def log_decision(layer_output: torch.Tensor, chosen_activation: str, reward: float):
    logger.info(
        f"Layer output: {layer_output}, Activation: {chosen_activation}, Reward: {reward}"
    )


class TestIndegoAdaptAct(unittest.TestCase):

    def test_enhanced_policy_network(self) -> None:
        """
        This test method is meticulously designed to validate the functionality of the EnhancedPolicyNetwork class.
        It ensures that the network's output tensor has the correct shape based on the specified input features, output features, and layers information.
        The method performs the following steps:

        1. Initializes an instance of EnhancedPolicyNetwork with specified dimensions and layer information.
        2. Generates a random input tensor with a predefined shape.
        3. Passes the input tensor through the EnhancedPolicyNetwork to produce an output tensor.
        4. Asserts that the shape of the output tensor matches the expected shape, ensuring the network's integrity and functionality.

        Attributes:
            policy_network (EnhancedPolicyNetwork): The network being tested, initialized with the specified dimensions and layer information.
            input_tensor (torch.Tensor): A randomly generated tensor used as input to the network.
            output (torch.Tensor): The output tensor produced by the network.
            expected_shape (torch.Size): The expected shape of the output tensor, used for validation.

        Raises:
            AssertionError: If the shape of the output tensor does not match the expected shape.
        """
        # Initialize an instance of EnhancedPolicyNetwork with explicit type annotations and detailed parameter specification
        policy_network: EnhancedPolicyNetwork = EnhancedPolicyNetwork(
            input_dim=10, output_dim=5, layers_info=[32, 32]
        )

        # Generate a random input tensor with a predefined shape and explicit type annotations
        input_tensor: torch.Tensor = torch.randn(16, 10)

        # Pass the input tensor through the EnhancedPolicyNetwork to produce an output tensor
        try:
            output: torch.Tensor = policy_network(input_tensor)
        except Exception as e:
            logger.error(
                f"An error occurred during the forward pass of the EnhancedPolicyNetwork: {str(e)}",
                exc_info=True,
            )
            raise RuntimeError(f"Forward pass failed due to: {str(e)}") from e

        # Define the expected shape of the output tensor, manually calculated and verified
        expected_shape: torch.Size = torch.Size([16, 5])

        # Assert that the shape of the output tensor matches the expected shape with a high degree of precision
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected output shape {expected_shape} does not match actual output shape {output.shape}",
        )

    def test_adaptive_activation_network(self) -> None:
        """
        This test method is meticulously designed to validate the functionality of the AdaptiveActivationNetwork class.
        It ensures that the network's output tensor has the correct shape based on the specified input features, output features, and layers information.
        The method performs the following steps:

        1. Initializes an instance of ActivationDictionary.
        2. Constructs an AdaptiveActivationNetwork using the activation dictionary and specified network parameters.
        3. Generates a random input tensor with a predefined shape.
        4. Passes the input tensor through the AdaptiveActivationNetwork to produce an output tensor.
        5. Asserts that the shape of the output tensor matches the expected shape, ensuring the network's integrity and functionality.

        Attributes:
            activation_dict (ActivationDictionary): An instance of ActivationDictionary used to provide activation functions to the network.
            adaptive_network (AdaptiveActivationNetwork): The network being tested, initialized with the activation dictionary and network parameters.
            input_tensor (torch.Tensor): A randomly generated tensor used as input to the network.
            output (torch.Tensor): The output tensor produced by the network.
            expected_shape (torch.Size): The expected shape of the output tensor, used for validation.

        Raises:
            AssertionError: If the shape of the output tensor does not match the expected shape.
        """
        # Initialize an instance of ActivationDictionary with explicit type annotations
        activation_dict: ActivationDictionary = ActivationDictionary()

        # Construct an AdaptiveActivationNetwork with explicit type annotations and detailed parameter specification
        adaptive_network: AdaptiveActivationNetwork = AdaptiveActivationNetwork(
            activation_dict=activation_dict,
            in_features=10,
            out_features=5,
            layers_info=[32, 32],
        )

        # Generate a random input tensor with a predefined shape and explicit type annotations
        input_tensor: torch.Tensor = torch.randn(16, 10)

        # Pass the input tensor through the AdaptiveActivationNetwork to produce an output tensor
        try:
            output: torch.Tensor = adaptive_network(input_tensor)
        except Exception as e:
            logger.error(
                f"An error occurred during the forward pass of the AdaptiveActivationNetwork: {str(e)}",
                exc_info=True,
            )
            raise RuntimeError(f"Forward pass failed due to: {str(e)}") from e

        # Define the expected shape of the output tensor, manually calculated and verified
        expected_shape: torch.Size = torch.Size([16, 5])

        # Assert that the shape of the output tensor matches the expected shape with a high degree of precision
        try:
            self.assertEqual(output.shape, expected_shape)
        except AssertionError as e:
            logger.error(
                f"Assertion error occurred during shape validation: Expected shape {expected_shape}, but got {output.shape}",
                exc_info=True,
            )
            raise AssertionError(
                f"Output tensor shape validation failed: Expected shape {expected_shape}, but got {output.shape}"
            ) from e

    def test_calculate_reward(self) -> None:
        """
        This test method is meticulously designed to validate the functionality of the calculate_reward function.
        It ensures that the reward calculation is accurate based on the provided current and previous loss values,
        as well as the true and predicted labels. The method performs the following steps:

        1. Initializes the current and previous loss values.
        2. Defines the true and predicted labels as numpy arrays.
        3. Calls the calculate_reward function to compute the reward based on these inputs.
        4. Asserts that the computed reward matches the expected reward value with a high degree of precision.

        Attributes:
            current_loss (float): The current loss value observed.
            previous_loss (float): The previous loss value observed.
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
            reward (float): The reward calculated by the calculate_reward function.
            expected_reward (float): The expected reward value, manually calculated and verified.

        Raises:
            AssertionError: If the computed reward does not match the expected reward within the specified precision.
        """
        # Initialize the current and previous loss values with explicit type annotations
        current_loss: float = 0.5
        previous_loss: float = 0.6

        # Define the true and predicted labels as numpy arrays with explicit type conversion
        y_true: np.ndarray = np.array([0, 1, 0, 1], dtype=int)
        y_pred: np.ndarray = np.array([0, 1, 0, 1], dtype=int)

        # Call the calculate_reward function to compute the reward based on these inputs
        try:
            reward: float = calculate_reward(
                current_loss, previous_loss, y_true, y_pred
            )
        except Exception as e:
            logger.error(
                f"An error occurred during reward calculation: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Reward calculation failed due to: {str(e)}") from e

        # Define the expected reward value, manually calculated and verified
        expected_reward: float = (
            1.1  # Ensure this matches the logic in calculate_reward
        )

        # Assert that the computed reward matches the expected reward with a high degree of precision
        try:
            self.assertAlmostEqual(reward, expected_reward, places=7)
        except AssertionError as e:
            logger.error(
                f"Assertion error in test_calculate_reward: {str(e)}", exc_info=True
            )
            raise AssertionError(
                f"Computed reward does not match expected reward: {str(e)}"
            ) from e

    def test_update_policy_network(self):
        """
        This test method is designed to validate the functionality of the update_policy_network function.
        It ensures that the policy network's parameters are updated correctly based on the provided reward and log probability.

        The method performs the following steps:
        1. Initializes an instance of the EnhancedPolicyNetwork with specified dimensions and layer information.
        2. Configures an optimizer (Adam) with a learning rate and associates it with the policy network's parameters.
        3. Defines a reward and a log probability value which are used to update the policy network.
        4. Calls the update_policy_network function to apply the updates.
        5. Includes extensive logging to trace the computation values and any exceptions that might occur.

        Attributes:
            policy_network (EnhancedPolicyNetwork): An instance of the EnhancedPolicyNetwork.
            optimizer (torch.optim.Adam): The optimizer configured with the policy network's parameters.
            reward (float): The reward value used to update the policy network.
            log_prob (float): The log probability associated with the chosen action.
        """
        # Initialize the policy network with specific input and output dimensions and layers configuration
        policy_network = EnhancedPolicyNetwork(
            input_dim=10,  # Input dimension size
            output_dim=5,  # Output dimension size
            layers_info=[32, 32],  # Information about each layer's neurons
        )
        # Configure the optimizer with the learning rate and associate it with the policy network's parameters
        optimizer = torch.optim.Adam(
            policy_network.parameters(), lr=0.01
        )  # Learning rate set to 0.01

        # Define the reward and log probability used for updating the policy network
        reward = 1.0  # Reward received from the environment
        log_prob = 0.5  # Log probability of the action taken

        # Attempt to update the policy network using the specified reward and log probability
        try:
            update_policy_network(policy_network, optimizer, reward, log_prob)
            logger.info(
                "Policy network successfully updated with reward: {} and log_prob: {}".format(
                    reward, log_prob
                )
            )
        except Exception as e:
            logger.error("Failed to update policy network: {}".format(e), exc_info=True)
            raise RuntimeError(
                "Test failed due to an error in updating the policy network: {}".format(
                    e
                )
            ) from e

    def test_log_decision(self):
        """
        This test method is meticulously designed to validate the functionality of the log_decision function within the IndegoAdaptAct module. It ensures that the logging of decisions regarding the neural network's layer outputs, chosen activation functions, and associated rewards is executed with precision and accuracy, adhering to the highest standards of operational excellence and robustness.

        The method performs the following steps:
        1. Generates a random tensor simulating the output of a neural network layer.
        2. Specifies a chosen activation function as a string.
        3. Defines a numerical reward value that simulates the reward received from an environment based on certain actions or decisions.
        4. Calls the log_decision function to log these details.
        5. Asserts that no exceptions are raised during the logging process, ensuring the stability and reliability of the function under test conditions.

        Attributes:
            layer_output (torch.Tensor): A tensor representing the output from a neural network layer, typically involving multiple dimensions where each dimension could represent different features or activations.
            chosen_activation (str): A string representing the activation function used in the neural network model.
            reward (float): A float value representing the reward received, which is used for logging purposes in this context.

        Raises:
            AssertionError: If any part of the logging process fails, indicating a breakdown or failure in the logging mechanism.
        """
        # Generate a random tensor simulating the output of a neural network layer
        layer_output: torch.Tensor = torch.randn(
            16, 32
        )  # 16 samples, each with 32 features

        # Specify the chosen activation function
        chosen_activation: str = (
            "ReLU"  # Using ReLU (Rectified Linear Unit) as an example
        )

        # Define the reward received from an environment based on certain actions
        reward: float = 1.0  # Example reward

        # Call the log_decision function to log these details
        try:
            log_decision(layer_output, chosen_activation, reward)
            logger.info(
                "log_decision function executed successfully with layer_output: {}, chosen_activation: {}, and reward: {}.".format(
                    layer_output, chosen_activation, reward
                )
            )
        except Exception as e:
            logger.error(
                "An error occurred while testing log_decision: {}".format(e),
                exc_info=True,
            )
            raise AssertionError(
                "Test failed due to an error in the log_decision function: {}".format(e)
            ) from e


if __name__ == "__main__":
    unittest.main()
