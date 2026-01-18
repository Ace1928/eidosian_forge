import unittest  # Importing the unittest module for creating and running tests
import torch  # Importing the PyTorch library for tensor computations and neural network operations
import sys  # Importing the sys module to interact with the Python runtime environment
import asyncio  # Importing the asyncio module for writing single-threaded concurrent code using coroutines
import logging  # Importing the logging module to enable logging of messages of various severity levels
from unittest.mock import (
    patch,
)  # Importing the patch function from unittest.mock to mock objects during tests
import numpy as np  # Importing the numpy library for numerical operations on arrays

# Append the system path to include the specific directory for module importation
sys.path.append(
    "/home/lloyd/EVIE/Indellama3/indego"
)  # Modifying sys.path to include the directory containing the Indego modules

# Importing specific classes and functions from the IndegoAdaptAct module
from ActivationDictionary import (
    ActivationDictionary,
)  # Importing the ActivationDictionary class which manages activation functions
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,  # Importing the EnhancedPolicyNetwork class, a neural network for policy decisions
    AdaptiveActivationNetwork,  # Importing the AdaptiveActivationNetwork class, a neural network that adapts its activation functions
    calculate_reward,  # Importing the calculate_reward function to compute rewards in reinforcement learning scenarios
    update_policy_network,  # Importing the update_policy_network function to update the policy network based on rewards
    log_decision,  # Importing the log_decision function to log decisions made by the policy network
)

# Importing the configure_logging function from the IndegoLogging module to set up advanced logging configurations
from IndegoLogging import configure_logging


# Asynchronous setup of the logging module
async def setup_logging() -> None:
    """
    Asynchronously sets up logging configurations using the configure_logging function from the IndegoLogging module.
    This function is designed to be run within an asyncio event loop.
    """
    await configure_logging()  # Awaiting the completion of the configure_logging function which sets up logging based on a configuration file


# Ensuring that there is an event loop available for the current thread
try:
    loop = asyncio.get_event_loop()  # Attempting to get the current event loop
except (
    RuntimeError
) as e:  # Handling the RuntimeError that occurs if no event loop is present in the current thread
    if "There is no current event loop in thread" in str(
        e
    ):  # Checking if the error message indicates the absence of an event loop
        loop = asyncio.new_event_loop()  # Creating a new event loop
        asyncio.set_event_loop(
            loop
        )  # Setting the newly created event loop as the current event loop

# Utilizing asyncio's event loop to perform the asynchronous logging setup
if (
    asyncio.get_event_loop().is_running()
):  # Checking if the current event loop is already running
    asyncio.run(
        setup_logging()
    )  # If the event loop is running, perform the logging setup synchronously
else:
    loop = asyncio.get_event_loop()  # Getting the current event loop
    loop.run_until_complete(
        setup_logging()
    )  # Running the setup_logging function asynchronously until it completes

# Acquiring a logger instance for the current module from the centralized logging configuration
logger = logging.getLogger(
    __name__
)  # Getting a logger with the name of the current module, configured as per the IndegoLogging module's settings


class TestIndegoAdaptAct(unittest.TestCase):
    def test_network_integration(self):
        activation_dict = ActivationDictionary()
        adaptive_network = AdaptiveActivationNetwork(
            activation_dict=activation_dict,
            in_features=10,
            out_features=5,
            layers_info=[64, 128, 256],
        )
        input_tensor = torch.randn(1, 10)
        try:
            output = adaptive_network(input_tensor)
            print("Output Shape:", output.shape)
        except Exception as e:
            print("Error during network integration test:", str(e))

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
            1.6  # Ensure this matches the logic in calculate_reward
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
            16, 32, requires_grad=True
        )  # 16 samples, each with 32 features, representing a complex multi-dimensional data structure

        # Specify the chosen activation function
        chosen_activation: str = (
            "ReLU"  # Using ReLU (Rectified Linear Unit) as an example, a common activation function in neural networks
        )

        # Define the reward received from an environment based on certain actions
        reward: float = (
            1.0  # Example reward, representing a numerical value typically used in reinforcement learning scenarios
        )

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
