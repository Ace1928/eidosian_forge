import math
import logging
import cmath  # Importing cmath for handling complex mathematical operations


class ActivationFunctionManager:
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

        Each lambda function is defined with explicit use of mathematical operations to ensure clarity and precision in their implementation.
        """
        self.activation_types = {
            "ReLU": lambda x: (
                max(0, x.real) + max(0, x.imag) * 1j
                if isinstance(x, complex)
                else max(0, x)
            ),
            "Sigmoid": lambda x: (
                1 / (1 + cmath.exp(-x))
                if x.real >= 0
                else cmath.exp(x) / (1 + cmath.exp(x))
            ),
            "Tanh": lambda x: (
                cmath.tanh(x) if abs(x) < 20 else (1 if x.real > 0 else -1)
            ),  # Handling extreme values for stability in deep networks
        }

        # Log the initialization of each activation function with detailed mathematical descriptions and expected input-output ranges.
        logging.debug(
            "Initialized ReLU activation function: f(x) = max(0, x) for x in real numbers, extended to complex numbers."
        )
        logging.debug(
            "Initialized Sigmoid activation function: f(x) = 1 / (1 + exp(-x)) for x >= 0, otherwise exp(x) / (1 + exp(x)) to maintain numerical stability."
        )
        logging.debug(
            "Initialized Tanh activation function: f(x) = tanh(x) for |x| < 20, otherwise 1 or -1 based on the sign of x to prevent overflow in deep networks."
        )

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
