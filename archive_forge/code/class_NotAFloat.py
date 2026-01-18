import unittest
class NotAFloat:

    def __float__(self):
        raise RuntimeError('I am not a float')