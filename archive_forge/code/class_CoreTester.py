import unittest
import torch
from trl.core import masked_mean, masked_var, masked_whiten, whiten
class CoreTester(unittest.TestCase):
    """
    A wrapper class for testing core utils functions
    """

    @classmethod
    def setUpClass(cls):
        cls.test_input = torch.Tensor([1, 2, 3, 4])
        cls.test_mask = torch.Tensor([0, 1, 1, 0])
        cls.test_input_unmasked = cls.test_input[1:3]

    def test_masked_mean(self):
        assert torch.mean(self.test_input_unmasked) == masked_mean(self.test_input, self.test_mask)

    def test_masked_var(self):
        assert torch.var(self.test_input_unmasked) == masked_var(self.test_input, self.test_mask)

    def test_masked_whiten(self):
        whiten_unmasked = whiten(self.test_input_unmasked)
        whiten_masked = masked_whiten(self.test_input, self.test_mask)[1:3]
        diffs = (whiten_unmasked - whiten_masked).sum()
        assert abs(diffs.item()) < 1e-05