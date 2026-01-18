import unittest
import torch
from ActivationDictionary import ActivationDictionary


class ActivationDictionaryTest(unittest.TestCase):

    def test_init(self):
        activation_dictionary = ActivationDictionary()
        self.assertEqual(len(activation_dictionary.activation_types), 32)

    def test_get_activation_function(self):
        activation_dictionary = ActivationDictionary()
        activation_function = activation_dictionary.get_activation_function("ReLU")
        self.assertIsNotNone(activation_function)
        self.assertEqual(activation_function(torch.tensor([1.0])), torch.tensor([1.0]))

        with self.assertRaises(KeyError):
            activation_dictionary.get_activation_function("NonExistentActivation")


if __name__ == "__main__":
    unittest.main()
