import logging
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable


class IndegoActivation:
    """
    Manages the activation functions within neural networks, ensuring a comprehensive and robust selection tailored to various network layers. This class encapsulates the complexity of activation function dynamics and provides a systematic approach to their management and application, adhering to the highest standards of software engineering and mathematical precision.

    Attributes:
        activation_types (dict): A dictionary mapping activation function names to their mathematical representations, allowing for dynamic selection and application.
        current_activation (str): The currently active activation function type, facilitating state tracking and operational consistency.

    Methods:
        initialize_activation_types(): Initializes the dictionary of activation functions with their respective mathematical implementations.
        apply_function(type: str, input: float) -> float: Applies the specified activation function to the input and returns the result, incorporating detailed logging and error handling.
    """

    def __init__(self):
        """
        Constructs an instance of the ActivationFunctionManager, meticulously setting up the foundational state and structure for managing a diverse array of activation functions utilized within neural networks. This constructor method is responsible for initializing the dictionary of activation functions, which encapsulates a variety of mathematical models tailored for neural computation. Additionally, it sets the initial state of the current activation to None, ensuring a clean slate for subsequent operations. This method adheres to the highest standards of software engineering, providing a robust and systematic approach to activation function management.

        The method performs the following operations:
        1. It calls the initialize_activation_types method to populate the activation_types dictionary with the respective lambda expressions representing the mathematical logic of each activation function.
        2. It initializes the current_activation attribute to None, establishing a neutral starting point for activation function application.
        3. It logs detailed information about the initialization process, specifically listing the supported activation function types, which enhances traceability and debugging capabilities.
        """
        self.initialize_activation_types()  # Populate the activation_types dictionary with mathematical implementations

        self.current_activation = None  # Initialize the current activation to None indicating no active function

        logging.info(
            "ActivationFunctionManager initialized with supported types: "
            + ", ".join(self.activation_types.keys())
        )

    def initialize_activation_types(self):
        """
        Establishes the dictionary of activation functions with their respective lambda expressions, meticulously encapsulating the mathematical logic required for each function. This method not only enhances modularity and maintainability of the activation function management but also ensures that each function is represented with the highest level of mathematical and computational integrity.

        The method performs the following detailed steps:
        1. It defines a lambda expression for the Rectified Linear Unit (ReLU) function, which applies a threshold operation that sets all negative inputs to zero, a critical operation for non-linear transformation in neural networks.
        2. It defines a lambda expression for the Sigmoid function, which maps the input values into a bounded range of [0, 1], serving as a smooth and differentiable approximation of a threshold mechanism and is widely used for binary classification problems.
        3. It defines a lambda expression for the Hyperbolic Tangent (Tanh) function, which also produces outputs in the range [-1, 1], effectively scaling the data within this interval and is particularly useful for modeling data that has been normalized to have zero mean and unit variance.
        4. It defines a lambda expression for the Softmax function, which is commonly used in the output layer of neural networks to normalize the output values into a probability distribution, facilitating multi-class classification tasks.
        5. It defines a lambda expression for the Linear function, which simply passes the input values through without any transformation, making it suitable for regression tasks where the output is expected to be a linear combination of the inputs.
        6. It defines a lambda expression for the Exponential Linear Unit (ELU) function, which introduces a non-zero gradient for negative inputs, addressing the vanishing gradient problem and providing improved learning dynamics in deep neural networks.
        7. It defines a lambda expression for the Swish function, which is a self-gated activation function that adapts the input-dependent scaling of the sigmoid function, offering a good balance between computational efficiency and representational power.
        8. It defines a lambda expression for the Leaky ReLU function, which allows a small gradient for negative inputs, preventing the dying ReLU problem and enabling the flow of gradients during backpropagation.
        9. It defines a lambda expression for the Parametric ReLU function, which introduces a learnable parameter to control the slope of the negative part of the function, providing flexibility in modeling the activation behavior.
        10. It defines a lambda expression for the Exponential Linear Units with Parametric Adjustment (ELU-PA) function, which extends the ELU function by introducing a learnable parameter to adjust the negative slope, offering additional flexibility in capturing complex activation patterns.
        11. It defines a lambda expression for the GELU function, which is a Gaussian Error Linear Unit that approximates the Gaussian cumulative distribution function, providing a smooth and non-linear activation function that is particularly effective in transformer architectures.
        12. It defines a lambda expression for the Softplus function, which is a smooth approximation of the ReLU function that is differentiable everywhere, ensuring continuous gradients and stable optimization during training.
        13. It defines a lambda expression for the Softsign function, which scales the input values by their absolute sum, producing outputs in the range (-1, 1) and offering a smooth and differentiable activation function.
        14. It defines a lambda expression for the Bent Identity function, which is a non-linear activation function that introduces a slight curvature to the identity function, providing additional modeling capacity while maintaining linearity for most input values.
        15. It defines a lambda expression for the Hard Sigmoid function, which approximates the sigmoid function with a piecewise linear function, offering a computationally efficient alternative for applications where precision is not critical.

        Each lambda function is defined with explicit use of mathematical operations to ensure clarity and precision in their implementation.
        """

        def parallelized_computation(activation_dict):
            """
            Executes the initialization of activation functions in a parallelized manner using a ThreadPoolExecutor.
            This method ensures that each activation function is initialized concurrently, leveraging multi-threading
            to enhance performance and efficiency.

            Parameters:
                activation_dict (dict): A dictionary containing the activation function names and their corresponding
                                        lambda expressions for initialization.

            The method logs detailed information about each activation function as it is initialized, and captures
            and logs any exceptions that occur during the parallel execution process.
            """
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submitting tasks to the executor and mapping futures to activation function names
                future_to_activation = {
                    executor.submit(lambda x: activation_dict[x](), name): name
                    for name in activation_dict
                }
                for future in concurrent.futures.as_completed(future_to_activation):
                    activation_name = future_to_activation[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        logging.error(
                            f"{activation_name} generated an exception: {exc}"
                        )
                    else:
                        logging.info(f"{activation_name}: Computation result - {data}")

        # Definition of activation functions using PyTorch operations, encapsulated in lambda expressions
        self.activation_types = {
            "ReLU": lambda x: torch.relu(torch.tensor(x, dtype=torch.complex64)).item(),
            "Sigmoid": lambda x: torch.sigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Tanh": lambda x: torch.tanh(torch.tensor(x, dtype=torch.complex64)).item(),
            "Softmax": lambda x: torch.softmax(
                torch.tensor([x], dtype=torch.complex64), dim=0
            ).tolist(),
            "Linear": lambda x: x,
            "ELU": lambda x: torch.nn.functional.elu(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Swish": lambda x: x
            * torch.sigmoid(torch.tensor(x, dtype=torch.complex64)).item(),
            "Leaky ReLU": lambda x: torch.nn.functional.leaky_relu(
                torch.tensor(x, dtype=torch.complex64), negative_slope=0.01
            ).item(),
            "Parametric ReLU": lambda x, a=0.01: torch.nn.functional.prelu(
                torch.tensor([x], dtype=torch.complex64), torch.tensor([a])
            ).item(),
            "ELU-PA": lambda x, a=0.01: torch.nn.functional.elu(
                torch.tensor(x, dtype=torch.complex64), alpha=a
            ).item(),
            "GELU": lambda x: torch.nn.functional.gelu(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Softplus": lambda x: torch.nn.functional.softplus(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Softsign": lambda x: torch.nn.functional.softsign(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Bent Identity": lambda x: (
                (torch.sqrt(torch.tensor(x, dtype=torch.complex64) ** 2 + 1) - 1) / 2
                + x
            ).item(),
            "Hard Sigmoid": lambda x: torch.nn.functional.hardsigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
        }

        # Logging the initialization of each activation function with detailed mathematical descriptions and expected input-output ranges.
        for func_name, func in self.activation_types.items():
            logging.debug(
                f"Initialized {func_name} activation function with lambda expression: {func}"
            )

        # Utilizing parallel computation to initialize and log activation functions
        parallelized_computation(self.activation_types)

    def apply_function(self, type: str, input: float) -> float:
        """
        Applies the specified activation function to the given input using advanced mathematical models. This method includes comprehensive error handling to ensure that only supported activation types are used, and it logs detailed information about the application process.

        Parameters:
            type (str): The type of activation function to apply. Must be one of the supported types defined in the activation_types dictionary.
            input (float): The input value to which the activation function will be applied.

        Returns:
            float: The output from the activation function, calculated using the appropriate mathematical model.

        Raises:
            ValueError: If the specified activation type is not supported, an error is logged and a ValueError is raised to prevent misuse of the function.
        """
        if type not in self.activation_types:
            error_message = f"Unsupported activation type '{type}'. Available types: {', '.join(self.activation_types.keys())}."
            logging.error(error_message)
            raise ValueError(error_message)

        logging.info(f"Applying {type} activation function to input: {input}")

        result = self.activation_types[type](input)

        self.current_activation = type

        logging.debug(
            f"{type} activation function applied to input {input}, resulting in output {result}"
        )

        return result
