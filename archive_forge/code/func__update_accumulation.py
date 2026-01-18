from numpy import sqrt
from .gradient_descent import GradientDescentOptimizer
def _update_accumulation(self, index, grad):
    """Update the moments.

        Args:
            index (int): the index of the argument to update
            grad (ndarray): the gradient for that trainable param
        """
    self.accumulation['fm'][index] = self.beta1 * self.accumulation['fm'][index] + (1 - self.beta1) * grad
    self.accumulation['sm'][index] = self.beta2 * self.accumulation['sm'][index] + (1 - self.beta2) * grad ** 2