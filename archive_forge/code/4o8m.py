import unittest
import torch
import sys

# Append the system path to include the specific directory for module importation
sys.path.append("/home/lloyd/EVIE/Indellama3/indego")
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,
    AdaptiveActivationNetwork,
    setup_logging,
)
from unittest.mock import patch
from ActivationDictionary import ActivationDictionary

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

    def test_initialization(self):
        # Assert the initialization parameters of the network
        self.assertEqual(self.network.in_features, self.in_features)
        self.assertEqual(self.network.output_dim, self.out_features)
        self.assertEqual(
            len(self.network.layers), len(self.layers_info) * 3 + 1
        )  # Each layer + ReLU + BatchNorm + Output layer

    def test_forward_pass(self):
        # Create a random input tensor and perform a forward pass
        input_tensor = torch.randn(1, self.in_features)
        output = self.network(input_tensor)
        # Assert the shape of the output tensor
        self.assertEqual(output.shape, torch.Size([1, self.out_features]))


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

    def test_initialization(self):
        # Assert the initialization parameters of the network
        self.assertEqual(len(self.network.activations), len(self.activation_dict))
        self.assertEqual(self.network.in_features, self.in_features)
        self.assertEqual(self.network.out_features, self.out_features)

    def test_forward_pass(self):
        # Create a random input tensor, ensure the network is in evaluation mode, and perform a forward pass
        input_tensor = torch.randn(1, self.in_features)
        self.network.eval()
        with torch.no_grad():
            output = self.network(input_tensor)
            # Assert the output is a tensor and check its shape
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertEqual(output.shape, torch.Size([1, self.out_features]))


if __name__ == "__main__":
    unittest.main()
