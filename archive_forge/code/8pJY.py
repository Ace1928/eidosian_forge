import unittest
import torch
import numpy as np
from IndegoAdaptAct import (
    EnhancedPolicyNetwork,
    AdaptiveActivationNetwork,
    calculate_reward,
    update_policy_network,
    log_decision,
)


class TestIndegoAdaptAct(unittest.TestCase):

    def test_enhanced_policy_network_initialization(self):
        input_dim = 5
        output_dim = 3
        layers_info = [10, 15]
        hyperparameters = {
            "learning_rate": 0.01,
            "regularization_factor": 0.001,
            "discount_factor": 0.99,
        }
        network = EnhancedPolicyNetwork(
            input_dim, output_dim, layers_info, hyperparameters
        )
        self.assertEqual(network.input_dim, input_dim)
        self.assertEqual(network.output_dim, output_dim)
        self.assertEqual(network.layers_info, layers_info)
        self.assertEqual(network.hyperparameters, hyperparameters)

    def test_enhanced_policy_network_forward_pass(self):
        input_dim = 5
        output_dim = 3
        layers_info = [10, 15]
        network = EnhancedPolicyNetwork(input_dim, output_dim, layers_info)
        input_tensor = torch.randn(16, input_dim)
        output_tensor = network(input_tensor)
        self.assertEqual(output_tensor.shape, (16, output_dim))

    def test_adaptive_activation_network_initialization(self):
        activation_dict = ActivationDictionary()
        in_features = 5
        out_features = 3
        layers_info = [10, 15]
        network = AdaptiveActivationNetwork(
            activation_dict, in_features, out_features, layers_info
        )
        self.assertEqual(network.activations, activation_dict.activation_types)
        self.assertEqual(
            network.activation_keys, list(activation_dict.activation_types.keys())
        )
        self.assertEqual(len(network.model), len(layers_info) + 1)
        self.assertEqual(
            network.policy_network.input_dim, in_features + sum(layers_info)
        )
        self.assertEqual(
            network.policy_network.output_dim, len(activation_dict.activation_types)
        )

    def test_adaptive_activation_network_forward_pass(self):
        activation_dict = ActivationDictionary()
        in_features = 5
        out_features = 3
        layers_info = [10, 15]
        network = AdaptiveActivationNetwork(
            activation_dict, in_features, out_features, layers_info
        )
        input_tensor = torch.randn(16, in_features)
        output_tensor = network(input_tensor)
        self.assertEqual(output_tensor.shape, (16, out_features))

    def test_calculate_reward(self):
        current_loss = 0.5
        previous_loss = 1.0
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        reward = calculate_reward(current_loss, previous_loss, y_true, y_pred)
        self.assertAlmostEqual(reward, 1.25)

    def test_update_policy_network(self):
        policy_network = EnhancedPolicyNetwork(5, 3, [10, 15])
        optimizer = optim.Adam(policy_network.parameters(), lr=0.01)
        reward = 1.0
        log_prob = torch.tensor(0.5, requires_grad=True)
        update_policy_network(policy_network, optimizer, reward, log_prob)

    def test_log_decision(self):
        layer_output = torch.tensor([0.1, 0.2, 0.3, 0.4])
        chosen_activation = "ReLU"
        reward = 1.0
        log_decision(layer_output, chosen_activation, reward)


if __name__ == "__main__":
    unittest.main()
