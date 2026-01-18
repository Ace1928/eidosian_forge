import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys


# Extending the system path to include the specific directory for module importation
# This is crucial for ensuring that the Python interpreter recognizes and can import modules
# from the specified directory, which is essential for modular programming and maintaining
# a clean project structure.
sys.path.append("/home/lloyd/EVIE/Indellama3/indego")

# Importing ActivationDictionary from a custom module.
# ActivationDictionary is presumably a class or a module that handles various activation functions
# which are critical in neural network operations, providing flexibility and modularity in defining
# custom activations.
from ActivationDictionary import ActivationDictionary

# Create an instance of ActivationDictionary
activation_dict_instance = ActivationDictionary()

# Importing configure_logging from a custom logging module.
# This function is responsible for setting up logging configurations, which is essential for
# tracking the flow of the program and debugging. Proper logging is crucial in a production environment
# to ensure that all actions are recorded for audit and troubleshooting purposes.
from IndegoLogging import configure_logging


# Definition of an asynchronous function to set up logging.
# This function is asynchronous, which means it is designed to handle I/O-bound and high-level
# structured network code. Asynchronous functions are a cornerstone of writing concurrent code
# in modern Python, particularly useful in I/O bound operations.
async def setup_logging() -> None:
    """
    Asynchronously sets up the logging configuration for the application.

    This function utilizes the configure_logging coroutine from the IndegoLogging module to set up
    the logging system. It is designed to be called asynchronously, allowing for non-blocking execution
    and enabling other tasks to run concurrently while the logging setup is in progress.

    The setup_logging function is crucial for initializing the logging system, which is essential for
    tracking the flow of the program, debugging, and ensuring that all actions are recorded for auditing
    and troubleshooting purposes in a production environment.

    Returns:
        None
    """
    # Awaiting the configuration of logging which is an asynchronous operation.
    # The await keyword is used to pause the execution of the setup_logging function
    # until the configure_logging coroutine is finished, which helps in preventing blocking
    # of the event loop, allowing other operations to run concurrently.
    await configure_logging()


# Attempt to retrieve the current event loop.
# The event loop is the core of every asyncio application, used to schedule asynchronous
# operations and handle their execution. It provides a foundation for asynchronous I/O, event
# handling, and other coroutine-based functionalities.
try:
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
except RuntimeError as e:
    # Handling the case where no event loop is found for the current thread.
    # This is a critical error handling part where a new event loop is created and set
    # for the current thread if the current thread does not already have one.
    # This ensures that asynchronous operations do not fail due to the absence of an event loop.
    if "There is no current event loop in thread" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Checking if the event loop is currently running.
# This condition is crucial to determine how to properly handle the setup of logging.
# If the event loop is running, it implies that the setup can be directly run using asyncio.run,
# otherwise, it should be scheduled to run using run_until_complete on the event loop.
if asyncio.get_event_loop().is_running():
    asyncio.run(setup_logging())
else:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup_logging())

# Initializing a logger object for this module.
# The getLogger function fetches or creates a logger named __name__, which is a built-in variable
# set to the name of the current module. This logger object is configured to track events in this module,
# and it adheres to the configurations set up by the configure_logging function.
# Setting up a logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DimensionSafeLinear(nn.Module):
    """
    A custom linear layer that ensures consistent input and output dimensions.

    This class extends the nn.Module class and provides a linear transformation layer with built-in
    dimension checking. It ensures that the input tensor has the expected dimensions before applying
    the linear transformation, raising a ValueError if there is a mismatch.

    The purpose of this class is to provide a safer and more robust linear layer that helps prevent
    dimension-related errors during the forward pass of a neural network. By checking the input dimensions
    against the expected values, it adds an extra layer of safety and aids in debugging dimension inconsistencies.

    Attributes:
        input_dim (int): The expected input dimension of the tensor.
        output_dim (int): The desired output dimension after the linear transformation.
        linear (nn.Linear): The underlying linear transformation layer.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the linear layer, ensuring consistent input dimensions.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initializes the DimensionSafeLinear layer with the specified input and output dimensions.

        This method sets up the DimensionSafeLinear layer by initializing the input and output dimensions
        and creating an instance of the nn.Linear layer with the corresponding dimensions.

        Parameters:
            input_dim (int): The expected input dimension of the tensor.
            output_dim (int): The desired output dimension after the linear transformation.
        """
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.linear: nn.Linear = nn.Linear(input_dim, output_dim)

        # Log the initialization of the DimensionSafeLinear layer
        logger.debug(
            f"Initialized DimensionSafeLinear layer with input dimension {input_dim} and output dimension {output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the DimensionSafeLinear layer.

        This method checks the input tensor's dimensions against the expected input dimension specified
        during initialization. If there is a mismatch, it raises a ValueError with a detailed error message.
        If the dimensions are consistent, it applies the linear transformation using the underlying nn.Linear layer.

        Parameters:
            x (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.

        Raises:
            ValueError: If the input tensor's dimensions do not match the expected input dimension.
        """
        # Log the input tensor's shape for debugging purposes
        logger.debug(f"Input tensor shape: {x.shape}")

        # Check if the input tensor has the expected dimensions
        if x.dim() != 2:
            raise ValueError(
                f"Expected input tensor to be 2-dimensional, but got {x.dim()} dimensions"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, but got {x.shape[1]}"
            )

        # Apply the linear transformation
        output: torch.Tensor = self.linear(x)

        # Log the output tensor's shape for debugging purposes
        logger.debug(f"Output tensor shape: {output.shape}")

        # Return the output tensor
        return output


class EnhancedPolicyNetwork(nn.Module):
    """
    A highly sophisticated neural network module designed for policy decision-making in complex environments. This class
    encapsulates a deep neural network with multiple layers, each followed by a ReLU activation and batch normalization
    for stability. The network's architecture and hyperparameters can be dynamically specified, allowing for extensive
    customization and optimization based on specific application needs.

    Attributes:
        input_dim (int): The dimensionality of the input feature space.
        output_dim (int): The number of output dimensions which typically corresponds to the number of actions in a policy network.
        layers_info (List[int]): A list detailing the number of neurons in each hidden layer.
        hyperparameters (Dict[str, float]): A dictionary containing hyperparameters for the network's operation such as
            learning rate, regularization factor, and discount factor, with sensible defaults provided.
        model (nn.Sequential): The sequential container that holds the layers of the network.

    Methods:
        forward(x: torch.Tensor, batch_id: Optional[int] = None, deterministic: bool = False) -> torch.Tensor:
            Conducts a forward pass through the network, optionally in a deterministic mode, using DimensionSafeLinear to ensure correct input dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers_info: List[int],
        hyperparameters: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the EnhancedPolicyNetwork with specified dimensions and hyperparameters.

        This constructor method meticulously sets up a neural network architecture based on the provided input dimension,
        output dimension, and information about the layers. It also configures the network with a set of hyperparameters,
        providing defaults if none are specified. The network is composed of linear layers, each followed by a ReLU activation
        function and batch normalization to ensure the stability of the network during training.

        Parameters:
            input_dim (int): The number of input features to the network.
            output_dim (int): The number of output features from the network.
            layers_info (List[int]): A list detailing the number of neurons in each hidden layer.
            hyperparameters (Optional[Dict[str, float]]): A dictionary containing hyperparameters for the network's operation such as
                learning rate, regularization factor, and discount factor, with sensible defaults provided if not specified.

        Raises:
            ValueError: If any of the dimensions or layer info is not properly defined.
        """
        # Call the constructor of the superclass (nn.Module)
        super(EnhancedPolicyNetwork, self).__init__()

        # Validate the input dimensions and layers information
        if input_dim <= 0:
            raise ValueError(
                f"Invalid input dimension provided: {input_dim}. Input dimension must be a positive integer."
            )
        if output_dim <= 0:
            raise ValueError(
                f"Invalid output dimension provided: {output_dim}. Output dimension must be a positive integer."
            )
        if not layers_info or any(dim <= 0 for dim in layers_info):
            raise ValueError(
                f"Invalid layers information provided: {layers_info}. All layer dimensions must be positive integers."
            )

        # Assign instance variables with type annotations
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.layers_info: List[int] = layers_info

        # Set hyperparameters with defaults using a ternary conditional operator
        self.hyperparameters: Dict[str, float] = (
            hyperparameters
            if hyperparameters is not None
            else {
                "learning_rate": 0.001,
                "regularization_factor": 0.01,
                "discount_factor": 0.99,
            }
        )

        # Initialize the layers of the network
        layers: List[nn.Module] = []
        current_dim: int = input_dim  # Start with the input dimension

        # Construct each layer based on the layers_info
        for index, dim in enumerate(layers_info):
            # Validate the dimension provided for the layer
            if dim <= 0:
                raise ValueError(
                    f"Invalid number of neurons specified for layer {index}: {dim}. Each layer must have a positive number of neurons."
                )

            # Create a DimensionSafeLinear layer
            linear_layer: nn.Module = DimensionSafeLinear(current_dim, dim)
            layers.append(linear_layer)

            # Add a ReLU activation layer
            relu_activation_layer: nn.Module = nn.ReLU()
            layers.append(relu_activation_layer)

            # Add a batch normalization layer
            batch_normalization_layer: nn.Module = nn.BatchNorm1d(dim)
            layers.append(batch_normalization_layer)

            # Update the current dimension to the new layer's output dimension
            current_dim = dim

        # Add the final DimensionSafeLinear output layer
        final_output_layer: nn.Module = DimensionSafeLinear(current_dim, output_dim)
        layers.append(final_output_layer)

        # Combine all layers into a sequential model
        self.model: nn.Sequential = nn.Sequential(*layers)

        # Log the initialization details
        logger.debug(
            f"EnhancedPolicyNetwork initialized with input dimension {input_dim}, output dimension {output_dim}, "
            f"layers {layers_info}, and initial hyperparameters {self.hyperparameters}"
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_id: Optional[int] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Conducts a forward pass through the network, optionally in a deterministic mode, using DimensionSafeLinear to ensure correct input dimensions.

        This method performs a forward pass of the input tensor through the layers of the EnhancedPolicyNetwork. It first ensures the
        correctness of the input dimensions using the DimensionSafeLinear layer. If the `deterministic` flag is set to True, the network
        operates in a deterministic mode by setting a fixed random seed, which is useful for testing and reproducibility. The method
        logs detailed information about the forward pass, including the batch ID, deterministic mode, intermediate and final outputs,
        and any errors that occur during the process.

        Parameters:
            x (torch.Tensor): The input tensor containing the feature data.
            batch_id (Optional[int]): An optional identifier for the batch being processed, used for logging.
            deterministic (bool): If True, the network operates in a deterministic mode, useful for testing and reproducibility.

        Returns:
            torch.Tensor: The output from the network, typically representing Q-values in a reinforcement learning context.

        Raises:
            RuntimeError: If an error occurs during the forward pass.
        """
        logger.debug(
            f"Forward pass of the EnhancedPolicyNetwork initiated with batch_id={batch_id} and deterministic={deterministic}."
        )
        try:
            # Ensuring dimension safety by wrapping the model's forward pass with DimensionSafeLinear
            dimension_safe_layer = DimensionSafeLinear(self.input_dim, self.output_dim)
            x = dimension_safe_layer(x)  # Ensuring input dimension correctness
            output: torch.Tensor = self.model(x)  # Forward pass through the network
            logger.debug(
                f"Intermediate output of the model (Q-values for each action) before any post-processing: {output}"
            )

            if deterministic:
                torch.manual_seed(0)
                logger.debug("Deterministic mode is enabled. Seed set to 0.")

            logger.debug(
                f"Final output of the model (Q-values for each action) after processing: {output}"
            )
        except Exception as e:
            logger.error(
                f"An error occurred during the forward pass: {e}", exc_info=True
            )
            raise RuntimeError(f"Forward pass failed due to: {e}") from e
        return output


class AdaptiveActivationNetwork(nn.Module):
    """
    A neural network module that dynamically selects activation functions during the forward pass based on a policy network's recommendations. This network is designed to adapt its activation functions to optimize performance for different types of input data dynamically.

    Attributes:
        activations (Dict[str, Callable[[torch.Tensor], torch.Tensor]]): A dictionary mapping activation function names to their corresponding callable functions.
        activation_keys (List[str]): A list of keys from the activations dictionary, used for indexing.
        layers (nn.ModuleList): A list of linear layers that transform the input data.
        policy_network (EnhancedPolicyNetwork): A policy network that determines the optimal activation function to use at each layer based on the current input.
        activation_path_log (defaultdict): A log that records the sequence of activation functions used during each forward pass, indexed by batch_id.

    Methods:
        forward(x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
            Processes the input tensor x through multiple layers and dynamically selected activation functions, returning the transformed tensor.
        dynamic_activation(x: torch.Tensor, batch_id: Optional[int], deterministic: bool = False) -> torch.Tensor:
            Dynamically selects and applies activation functions to the input tensor `x` based on the outputs of a policy network.
    """

    def __init__(
        self,
        activation_dict: ActivationDictionary,
        in_features: int,
        out_features: int,
        layers_info: List[int],
    ) -> None:
        """
        Initializes the AdaptiveActivationNetwork with specified configurations for layers and activation functions,
        ensuring that each layer is meticulously constructed based on the provided specifications and that the policy
        network is aligned with the output features of the last layer.

        Parameters:
            activation_dict (ActivationDictionary): An object containing a comprehensive mapping of activation function
                names to their implementations, facilitating dynamic selection during the network's operation.
            in_features (int): The number of input features to the network, defining the dimensionality of the input data.
            out_features (int): The number of output features from the network, specifying the dimensionality of the output data.
            layers_info (List[int]): A detailed list specifying the number of neurons in each hidden layer, which dictates
                the architecture of the neural network.

        Raises:
            ValueError: If the layers_info list is empty, indicating that no layers have been defined for the network.
        """
        super(AdaptiveActivationNetwork, self).__init__()

        # Validate the presence of at least one layer information to construct the network architecture.
        if not layers_info:
            error_message: str = (
                "The layers_info parameter must contain at least one element to define the network layers."
            )
            logger.error(f"Initialization error: {error_message}")
            raise ValueError(error_message)

        # Retrieve and map activation functions from the provided activation dictionary to ensure dynamic functionality.
        self.activations: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = (
            activation_dict.activation_types
        )
        if not self.activations:
            error_message: str = (
                "The activation_dict provided does not contain any activation functions."
            )
            logger.error(f"Initialization error: {error_message}")
            raise ValueError(error_message)

        # Extract the keys from the activations dictionary to facilitate dynamic selection during the forward pass.
        self.activation_keys: List[str] = list(self.activations.keys())
        if not self.activation_keys:
            error_message: str = (
                "No activation functions found within the activation_dict."
            )
            logger.error(f"Initialization error: {error_message}")
            raise ValueError(error_message)

        # Constructing the layers of the network based on the specified layers_info and ensuring correct input-output dimensions.
        self.layers: nn.ModuleList = nn.ModuleList()
        previous_layer_output_features: int = in_features

        # Iteratively construct each layer based on the layers_info and append the final output layer.
        for layer_index, current_layer_features in enumerate(
            layers_info + [out_features]
        ):
            # Create a linear layer with the appropriate input and output features.
            current_layer: nn.Linear = nn.Linear(
                previous_layer_output_features, current_layer_features
            )
            self.layers.append(current_layer)

            # Log the creation of each layer for debugging and verification purposes.
            logger.debug(
                f"Layer {layer_index} created with input features {previous_layer_output_features} and output features {current_layer_features}."
            )

            # Update the previous_layer_output_features to the current layer's output features for the next iteration.
            previous_layer_output_features = current_layer_features

        # Log the successful initialization of the network layers.
        logger.info(
            f"All network layers initialized successfully with total layers: {len(self.layers)}"
        )
        # Initializing the policy network which determines the optimal activation function for each layer.
        self.policy_network: EnhancedPolicyNetwork = EnhancedPolicyNetwork(
            input_dim=previous_layer_output_features,  # Ensuring the input dimension matches the output of the last layer.
            output_dim=len(
                self.activations
            ),  # The output dimension is set to the number of available activation functions.
            layers_info=[
                previous_layer_output_features,
                *layers_info,
                out_features,
            ],  # Ensuring that the layers are the correct dim for the policy network.
        )

        # Logging the initialization parameters for debugging and verification purposes.
        logger.debug(
            f"AdaptiveActivationNetwork initialized with the following parameters: "
            f"activation keys={self.activation_keys}, input features dimension={in_features}, "
            f"output features dimension={out_features}, layers information={layers_info}."
        )

        # A log to record the sequence of activation functions used during each forward pass, indexed by batch_id.
        self.activation_path_log: defaultdict = defaultdict(list)

    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """
        Conducts a forward pass through the network, dynamically selecting activation functions based on the policy network's output.

        This method orchestrates the sequential processing of the input tensor `x` through multiple layers of the network, applying a dynamically selected activation function at each step, except for the final layer. The selection of activation functions is governed by the policy network, which determines the most suitable activation function for each layer based on the current state of the tensor `x`.

        Parameters:
            x (torch.Tensor): The input tensor that will be processed through the network.
            batch_id (Optional[int]): An optional identifier for the batch being processed, which is utilized for logging purposes to track the progress and behavior of the batch through the network.

        Returns:
            torch.Tensor: The output tensor after it has been processed through all the network layers and the dynamically selected activation functions.

        Raises:
            RuntimeError: If an error occurs at any point during the forward pass, encapsulating the original exception for detailed error analysis.

        Detailed Workflow:
            1. Log the initiation of the forward pass with the batch identifier.
            2. Sequentially process the input tensor through all network layers except the last, applying the dynamically selected activation functions.
            3. Process the final layer without applying any dynamic activation function.
            4. Log the output after the final layer processing.
            5. Handle any exceptions that occur during the process, log the error, and raise a RuntimeError with detailed information.
        """
        logger.debug(
            f"Forward pass through AdaptiveActivationNetwork initiated with batch_id={batch_id}."
        )
        try:
            # Iterate through all layers except the last one to apply the respective dynamically selected activation functions.
            for layer_index, layer in enumerate(
                self.layers[:-1]
            ):  # Exclude the last layer for separate processing
                logger.debug(
                    f"Processing layer {layer_index} with input tensor dimensions: {x.shape}"
                )
                x = DimensionSafeLinear(layer.in_features, layer.out_features)(
                    x
                )  # Process the current layer with dimension safety
                logger.debug(
                    f"Output tensor dimensions after DimensionSafeLinear at layer {layer_index}: {x.shape}"
                )
                x = self.dynamic_activation(
                    x, batch_id
                )  # Apply dynamic activation after each layer
                logger.debug(
                    f"Output tensor dimensions after dynamic activation at layer {layer_index}: {x.shape}"
                )

            # Process the last layer without any dynamic activation to finalize the output tensor formation.
            final_layer_index = len(self.layers) - 1
            final_layer = self.layers[-1]
            logger.debug(
                f"Processing final layer {final_layer_index} with input tensor dimensions: {x.shape}"
            )
            x = DimensionSafeLinear(final_layer.in_features, final_layer.out_features)(
                x
            )
            logger.debug(
                f"Output tensor dimensions after final layer {final_layer_index}: {x.shape}"
            )

        except Exception as e:
            # Log the exception with detailed stack trace information for debugging purposes.
            logger.error(
                f"An error occurred during the forward pass at layer {layer_index if 'layer_index' in locals() else 'unknown'}: {e}",
                exc_info=True,
            )
            # Raise a RuntimeError to indicate a failure in the forward pass, providing the original exception context.
            raise RuntimeError(f"Forward pass failed due to: {e}") from e

        # Log the final output after dynamic activation for traceability and debugging.
        logger.debug(f"Final output after dynamic activation: {x}")

        # Return the final processed tensor.
        return x

    def dynamic_activation(
        self, x: torch.Tensor, batch_id: Optional[int], deterministic: bool = False
    ) -> torch.Tensor:
        """
        Dynamically selects and applies activation functions to the input tensor `x` based on the outputs of a policy network.

        This method utilizes a policy network to determine the activation functions to be applied to each element in the input tensor `x`.
        Depending on the `deterministic` flag, it either selects the most probable activation function or samples from a distribution of possible activations.

        Parameters:
            x (torch.Tensor): The input tensor to which activation functions will be applied.
            batch_id (Optional[int]): An optional batch identifier used for logging and tracking the processing of batches.
            deterministic (bool): A flag to determine the mode of activation function selection. If True, the function with the highest score is selected. Otherwise, the selection is probabilistic.

        Returns:
            torch.Tensor: The tensor resulting from applying the selected activation functions to the input tensor `x`.

        Raises:
            RuntimeError: If there is an inconsistency in tensor sizes during padding or any other operation within the function.

        Detailed Workflow:
            1. Compute the activation scores using the policy network.
            2. Determine the selection mode (deterministic or probabilistic) and select activation indices accordingly.
            3. Retrieve the corresponding activation functions from a predefined dictionary using the selected indices.
            4. Apply each selected activation function to the corresponding element of the input tensor `x`.
            5. Calculate the maximum size among the activated tensors to standardize their sizes.
            6. Pad tensors with fewer elements than the maximum size to ensure uniformity.
            7. Stack the uniformly sized tensors to form the final output tensor.
        """
        # Step 1: Compute the activation scores using the policy network
        activation_scores: torch.Tensor = self.policy_network(x)
        logger.debug(f"Activation scores computed: {activation_scores}")

        # Step 2: Select activation indices based on the deterministic flag
        if deterministic:
            selected_activation_idx: torch.Tensor = torch.argmax(
                activation_scores, dim=-1
            )
            logger.info(
                f"Deterministic selection of activation indices: {selected_activation_idx}"
            )
        else:
            activation_probs: torch.Tensor = F.softmax(activation_scores, dim=-1)
            selected_activation_idx: torch.Tensor = torch.multinomial(
                activation_probs, 1
            ).squeeze(0)
            logger.info(
                f"Probabilistic selection of activation indices: {selected_activation_idx}"
            )

        # Step 3: Retrieve the corresponding activation functions
        selected_activations: List[Callable[[torch.Tensor], torch.Tensor]] = [
            self.activations[self.activation_keys[idx]]
            for idx in selected_activation_idx
        ]
        logger.debug(f"Selected activation functions: {selected_activations}")

        # Step 4: Apply the activation functions using DimensionSafeLinear
        activated_tensors: List[torch.Tensor] = []
        for x_i, act in zip(x, selected_activations):
            dimension_safe_layer = DimensionSafeLinear(x_i.shape[0], x_i.shape[0])
            x_i = dimension_safe_layer(x_i)
            activated_tensors.append(act(x_i))

        # Calculate the maximum size among the activated tensors
        max_size: int = max(
            (tensor.nelement() for tensor in activated_tensors), default=0
        )

        padded_tensors: List[torch.Tensor] = []
        for tensor in activated_tensors:
            if tensor.nelement() == 0:
                # For empty tensors, create a new tensor of the desired shape filled with zeros
                if max_size > 0:
                    shape = (
                        max_size,
                    )  # Assuming 1D tensors for simplicity; adjust as needed
                    padded_tensor = torch.zeros(
                        shape, dtype=tensor.dtype, device=tensor.device
                    )
                else:
                    padded_tensor = torch.tensor(
                        [], dtype=tensor.dtype, device=tensor.device
                    )
            else:
                # Calculate padding needed to match the max_size
                padding_needed = max_size - tensor.nelement()
                # Adjust padding logic to ensure it's compatible with tensor dimensions
                # This example assumes 1D tensors; for multidimensional tensors, the padding logic will need to be adjusted accordingly
                padded_tensor = F.pad(tensor, (0, padding_needed))

            padded_tensors.append(padded_tensor)

        final_output = torch.stack(padded_tensors)
        return final_output


def calculate_reward(
    current_loss: float,
    previous_loss: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate the reward based on the improvement in loss and the performance metrics of the model.

    This function computes the reward for a model's prediction by considering both the improvement in loss
    and the combined performance metrics such as precision, recall, and F1 score. The reward is a weighted sum
    of the loss improvement and the average of the mentioned metrics.

    Parameters:
        current_loss (float): The loss measured after the latest model update.
        previous_loss (float): The loss measured prior to the latest model update.
        y_true (np.ndarray): The true labels of the data.
        y_pred (np.ndarray): The predicted labels by the model.

    Returns:
        float: The calculated reward value which is a float.

    Raises:
        ValueError: If the input arrays y_true and y_pred have different lengths.

    Detailed Workflow:
        1. Validate the input arrays for consistent length.
        2. Calculate the improvement in loss by subtracting the current loss from the previous loss.
        3. Log the calculated loss improvement for debugging purposes.
        4. Compute the precision, recall, and F1 score using the true and predicted labels.
        5. Log the computed metrics for detailed analysis and verification.
        6. Calculate the final reward by adding the loss improvement to half the sum of the precision,
           recall, and F1 score metrics.
        7. Log the final constructed reward for further debugging and verification.
        8. Return the computed reward.
    """
    # Step 1: Validate the input arrays for consistent length
    if y_true.shape[0] != y_pred.shape[0]:
        logger.error(
            "The true labels and predicted labels arrays do not match in length."
        )
        raise ValueError(
            "The length of true labels and predicted labels must be the same."
        )

    # Step 2: Calculate the improvement in loss
    loss_improvement: float = previous_loss - current_loss
    logger.debug(f"Calculated loss improvement: {loss_improvement}")

    # Step 3: Compute precision, recall, and F1 score
    precision_metric: float = precision_score(y_true, y_pred, average="macro")
    recall_metric: float = recall_score(y_true, y_pred, average="macro")
    f1_metric: float = f1_score(y_true, y_pred, average="macro")

    # Step 4: Log the computed metrics
    logger.info(
        f"Reward calculation details: Loss Improvement: {loss_improvement}, Precision: {precision_metric}, Recall: {recall_metric}, F1 Score: {f1_metric}"
    )

    # Step 5: Calculate the final reward
    reward: float = loss_improvement + 0.5 * (
        precision_metric + recall_metric + f1_metric
    )
    logger.debug(f"Constructed reward using weighted metrics: {reward}")

    # Step 6: Return the computed reward
    return reward


def update_policy_network(
    policy_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    reward: float,
    log_prob: torch.Tensor = torch.tensor(0.5, requires_grad=True),
) -> None:
    """
    This function meticulously updates the policy network by applying the policy gradient method, a cornerstone of reinforcement learning techniques. It calculates the policy loss based on the log probabilities of actions taken and the received reward, then performs backpropagation to update the weights of the policy network. This method is crucial for the iterative improvement of the policy network's performance in decision-making tasks.

    Parameters:
        policy_network (nn.Module): The neural network model that defines the policy, responsible for decision-making processes.
        optimizer (torch.optim.Optimizer): The optimizer used for applying gradient descent methods to update the network weights, facilitating the learning process.
        reward (float): The reward received from the environment after taking an action, serving as feedback for the policy's effectiveness.
        log_prob (torch.Tensor): The logarithm of the probability of the action taken, as output by the policy network, used in the calculation of the policy gradient.

    Raises:
        RuntimeError: If an unexpected error occurs during the update process, encapsulating the original exception for robust error handling.

    Detailed Workflow:
        1. Convert the scalar reward into a tensor for compatibility with PyTorch operations, ensuring type consistency.
        2. Calculate the policy loss as the negative product of the log probability and the reward tensor, adhering to the policy gradient formula.
        3. Reset the gradients of the optimizer to zero to prevent accumulation from previous iterations, ensuring a clean slate for gradient computation.
        4. Perform backpropagation to compute the gradients of the loss with respect to the network parameters, a critical step for learning.
        5. Update the weights of the network using the optimizer based on the computed gradients, applying the learning from the current iteration.
        6. Log detailed information about the operations performed, including the calculated loss and the status of the optimizer, for thorough monitoring and debugging.
    """
    try:
        # Step 1: Convert the scalar reward into a tensor to ensure compatibility with PyTorch operations.
        reward_tensor: torch.Tensor = torch.tensor(
            reward, requires_grad=True, dtype=torch.float32, device=log_prob.device
        )
        logger.debug(f"Converted reward to tensor: {reward_tensor}")

        # Step 2: Calculate the policy loss, which is the negative product of the log probability of the action and the reward.
        policy_loss: torch.Tensor = -log_prob * reward_tensor
        logger.debug(f"Calculated policy loss: {policy_loss.item()}")

        # Step 3: Reset the gradients of all optimized tensors to zero.
        optimizer.zero_grad()
        logger.debug("Reset optimizer gradients to zero.")

        # Step 4: Perform backpropagation to compute the gradients of the loss with respect to the network parameters.
        policy_loss.backward()
        logger.debug("Performed backpropagation to compute gradients.")

        # Step 5: Update the weights of the policy network based on the gradients computed during backpropagation.
        optimizer.step()
        logger.debug("Updated weights of the policy network.")

        # Step 6: Log the completion of the policy network update process.
        logger.debug(
            f"Completed policy network update. Computed Loss: {policy_loss.item()}"
        )

    except Exception as e:
        # Log the error with detailed exception information and re-raise a RuntimeError to indicate failure in the update process.
        logger.error(
            f"An error occurred during the policy network update: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Policy network update failed due to: {e}") from e


def log_decision(
    layer_output: torch.Tensor, chosen_activation: str, reward: float
) -> None:
    """
    Logs the decision details including the neural network layer output, the chosen activation function, and the reward received.

    This function meticulously captures and logs the decision-making process within the neural network, providing a detailed, structured, and traceable log entry that includes the output of a specific layer, the activation function applied, and the reward received as a result of the action taken. This detailed logging is crucial for debugging, monitoring, and analyzing the behavior of the neural network in response to various inputs and conditions.

    Parameters:
        layer_output (torch.Tensor): The output tensor from a neural network layer, which contains data that has been processed by the layer.
        chosen_activation (str): The name of the activation function that was applied to the layer output.
        reward (float): The numerical reward received from the environment, which quantifies the success of the action taken based on the network's output.

    Raises:
        RuntimeError: If an error occurs during the logging process, encapsulating the original exception for detailed error tracing.

    Detailed Workflow:
        1. Convert the layer output tensor to a list for readability in the log.
        2. Construct a formatted log message that includes all relevant decision details.
        3. Use the logging library to log the constructed message at the INFO level.
        4. In case of an exception, log the error at the ERROR level with detailed exception information and raise a RuntimeError to indicate a failure in the logging process.
    """
    try:
        # Step 1: Convert the tensor to a list for more human-readable form in logs
        layer_output_list: List[float] = layer_output.tolist()
        logger.debug(f"Converted layer output to list: {layer_output_list}")

        # Step 2: Construct the log message with detailed information
        log_message: str = (
            f"Decision Log - Layer Output: {layer_output_list}, "
            f"Chosen Activation Function: {chosen_activation}, "
            f"Reward Received: {reward}"
        )
        logger.debug(f"Constructed log message: {log_message}")

        # Step 3: Log the detailed decision information at the INFO level
        logger.info(log_message)

        # Step 4: Confirm the successful logging of the decision
        logger.debug("Successfully logged the decision details at the INFO level.")

    except Exception as e:
        # Step 5: Construct an error message that includes the exception details
        error_message: str = f"An error occurred while logging the decision: {e}"
        logger.error(error_message, exc_info=True)

        # Step 6: Raise a RuntimeError to indicate a failure in the logging process, including the original exception for traceability
        raise RuntimeError(f"Logging of the decision failed due to: {e}") from e


def main() -> None:
    """
    The main function serves as the entry point for the neural network execution. It initializes the AdaptiveActivationNetwork with specified configurations, processes an input tensor, and outputs the results. This function is meticulously designed to handle the initialization, execution, and output processes with high precision and detailed logging.

    Detailed Workflow:
        1. Initialize the AdaptiveActivationNetwork with specific activation dictionary, input features, output features, and layer information.
        2. Generate a random input tensor with predefined dimensions.
        3. Process the input tensor through the network to obtain the output.
        4. Print the output of the network to the console for verification and debugging purposes.

    Raises:
        Exception: Captures and logs any exceptions that might occur during the network initialization or execution, ensuring robust error handling.
    """
    try:
        # Step 1: Initialize the network with specific configurations
        network: AdaptiveActivationNetwork = AdaptiveActivationNetwork(
            activation_dict_instance,
            in_features=20,
            out_features=20,
            layers_info=[20, 30],
        )
        logger.info(
            "AdaptiveActivationNetwork initialized successfully with configurations: in_features=20, out_features=20, layers_info=[20, 30]"
        )

        # Step 2: Generate a random input tensor of size 20x20
        input_tensor: torch.Tensor = torch.randn(20, 20)
        logger.debug(f"Generated random input tensor with shape {input_tensor.shape}")

        # Step 3: Process the input tensor through the network
        output: torch.Tensor = network(input_tensor, batch_id=1)
        logger.info(
            f"Processed input tensor through the network, resulting in output tensor with shape {output.shape}"
        )

        # Step 4: Print the output of the network
        print("Output of the network:", output)

    except Exception as e:
        # Log the exception with detailed information and re-raise to indicate failure in the main function
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)
        raise Exception(f"Failed to execute main function due to: {e}") from e


if __name__ == "__main__":
    # Attempt to run the main function and handle any exceptions that might occur
    try:
        main()
    except Exception as e:
        logger.critical(
            f"Critical failure in running the main function: {e}", exc_info=True
        )
