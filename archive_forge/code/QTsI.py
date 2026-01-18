import unittest
import torch
import sys

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

# Append the system path to include the specific directory for module importation
sys.path.append("/home/lloyd/EVIE/Indellama3/indego")
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,
    AdaptiveActivationNetwork,
    setup_logging,
)
from unittest.mock import patch
from ActivationDictionary import ActivationDictionary

# Configure logging to ensure all operations are logged with maximum detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Instantiate the ActivationDictionary to access activation functions
activation_dict_instance = ActivationDictionary()


class TestEnhancedPolicyNetwork(unittest.TestCase):
    def setUp(self):
        # Define input and output features along with layer information
        self.in_features = 10
        self.out_features = 4
        self.layers_info = [20, 15, 10]
        # Access activation types directly from the ActivationDictionary instance
        self.activation_dict = activation_dict_instance.activation_types
        # Initialize the AdaptiveActivationNetwork with the specified parameters
        self.network = AdaptiveActivationNetwork(
            self.activation_dict, self.in_features, self.out_features, self.layers_info
        )
        logging.info(
            "AdaptiveActivationNetwork initialized with in_features={}, out_features={}, layers_info={}".format(
                self.in_features, self.out_features, self.layers_info
            )
        )

    def test_initialization(self):
        # Assert the initialization parameters of the network
        self.assertEqual(self.network.in_features, self.in_features)
        self.assertEqual(self.network.output_dim, self.out_features)
        self.assertEqual(
            len(self.network.layers), len(self.layers_info) * 3 + 1
        )  # Each layer + ReLU + BatchNorm + Output layer
        logging.debug(
            "Initialization parameters asserted successfully for in_features, output_dim, and layers."
        )

    def test_forward_pass(self):
        # Create a random input tensor and perform a forward pass
        input_tensor = torch.randn(1, self.in_features)
        output = self.network(input_tensor)
        # Assert the shape of the output tensor
        self.assertEqual(output.shape, torch.Size([1, self.out_features]))
        logging.debug(
            "Forward pass successful with output shape: {}".format(output.shape)
        )


class TestAdaptiveActivationNetwork(unittest.TestCase):
    def setUp(self):
        # Define input and output features along with layer information
        self.in_features = 10
        self.out_features = 4
        self.layers_info = [20, 15, 10]
        # Access activation types directly from the ActivationDictionary instance
        self.activation_dict = activation_dict_instance
        # Initialize the AdaptiveActivationNetwork with the specified parameters
        self.network = AdaptiveActivationNetwork(
            self.activation_dict, self.in_features, self.out_features, self.layers_info
        )
        logging.info(
            "AdaptiveActivationNetwork initialized with in_features={}, out_features={}, layers_info={}".format(
                self.in_features, self.out_features, self.layers_info
            )
        )

    def test_initialization(self):
        # Assert the initialization parameters of the network
        self.assertEqual(len(self.network.activations), len(self.activation_dict))
        self.assertEqual(self.network.in_features, self.in_features)
        self.assertEqual(self.network.out_features, self.out_features)
        logging.debug(
            "Initialization parameters asserted successfully for activations, in_features, and out_features."
        )

    def test_forward_pass(self):
        # Create a random input tensor, ensure the network is in evaluation mode, and perform a forward pass
        input_tensor = torch.randn(1, self.in_features)
        self.network.eval()
        with torch.no_grad():
            output = self.network(input_tensor)
            # Assert the output is a tensor and check its shape
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertEqual(output.shape, torch.Size([1, self.out_features]))
            logging.debug(
                "Forward pass successful with output shape: {}".format(output.shape)
            )


if __name__ == "__main__":
    unittest.main()
