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
