import unittest
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict, Any, Optional


class IndegoActivationTest(unittest.TestCase):

    def test_initialize_activation_types(self):
        activation_manager = IndegoActivation()
        self.assertEqual(len(activation_manager.activation_types), 6)
        self.assertIn("relu", activation_manager.activation_types)
        self.assertIn("sigmoid", activation_manager.activation_types)
        self.assertIn("tanh", activation_manager.activation_types)
        self.assertIn("leaky_relu", activation_manager.activation_types)
        self.assertIn("elu", activation_manager.activation_types)
        self.assertIn("gelu", activation_manager.activation_types)

    def test_apply_function(self):
        activation_manager = IndegoActivation()
        input = torch.tensor(0.5)
        output = activation_manager.apply_function("relu", input)
        self.assertEqual(output, 0.5)
        output = activation_manager.apply_function("sigmoid", input)
        self.assertEqual(output, 0.5)
        output = activation_manager.apply_function("tanh", input)
        self.assertEqual(output, 0.46211716)
        output = activation_manager.apply_function("leaky_relu", input)
        self.assertEqual(output, 0.5)
        output = activation_manager.apply_function("elu", input)
        self.assertEqual(output, 0.5)
        output = activation_manager.apply_function("gelu", input)
        self.assertEqual(output, 0.5)

    def test_apply_function_error(self):
        activation_manager = IndegoActivation()
        input = torch.tensor(0.5)
        with self.assertRaises(ValueError):
            activation_manager.apply_function("invalid_activation", input)
