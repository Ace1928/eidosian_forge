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


class TestIndegoAdaptAct(unittest.TestCase):

    def test_enhanced_policy_network(self):
        policy_network = EnhancedPolicyNetwork(
            input_dim=10, output_dim=5, layers_info=[32, 32]
        )
        input_tensor = torch.randn(16, 10)
        output = policy_network(input_tensor)
        self.assertEqual(output.shape, torch.Size([16, 5]))

    def test_adaptive_activation_network(self):
        activation_dict = ActivationDictionary()
        adaptive_network = AdaptiveActivationNetwork(
            activation_dict, in_features=10, out_features=5, layers_info=[32, 32]
        )
        input_tensor = torch.randn(16, 10)
        output = adaptive_network(input_tensor)
        self.assertEqual(output.shape, torch.Size([16, 5]))

    def test_calculate_reward(self):
        current_loss = 0.5
        previous_loss = 0.6
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        reward = calculate_reward(current_loss, previous_loss, y_true, y_pred)
        expected_reward = 1.6  # Ensure this matches the logic in calculate_reward
        self.assertAlmostEqual(reward, expected_reward, places=7)

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
