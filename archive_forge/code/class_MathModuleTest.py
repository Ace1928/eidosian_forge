import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
class MathModuleTest(unittest.TestCase):
    """Math module tests."""

    def test_lerp(self):
        result = pygame.math.lerp(10, 100, 0.5)
        self.assertAlmostEqual(result, 55.0)
        result = pygame.math.lerp(10, 100, 0.0)
        self.assertAlmostEqual(result, 10.0)
        result = pygame.math.lerp(10, 100, 1.0)
        self.assertAlmostEqual(result, 100.0)
        self.assertRaises(TypeError, pygame.math.lerp, 1)
        self.assertRaises(TypeError, pygame.math.lerp, 'str', 'str', 'str')
        self.assertRaises(ValueError, pygame.math.lerp, 10, 100, 1.1)
        self.assertRaises(ValueError, pygame.math.lerp, 10, 100, -0.5)

    def test_clamp(self):
        """Test clamp function."""
        result = pygame.math.clamp(10, 1, 5)
        self.assertEqual(result, 5)
        result = pygame.math.clamp(-10, 1, 5)
        self.assertEqual(result, 1)
        result = pygame.math.clamp(5, 1, 5)
        self.assertEqual(result, 5)
        result = pygame.math.clamp(1, 1, 5)
        self.assertEqual(result, 1)
        result = pygame.math.clamp(3, 1, 5)
        self.assertEqual(result, 3)
        result = pygame.math.clamp(10.0, 1.12, 5.0)
        self.assertAlmostEqual(result, 5.0)
        result = pygame.math.clamp(-10.0, 1.12, 5.0)
        self.assertAlmostEqual(result, 1.12)
        result = pygame.math.clamp(5.0, 1.12, 5.0)
        self.assertAlmostEqual(result, 5.0)
        result = pygame.math.clamp(1.12, 1.12, 5.0)
        self.assertAlmostEqual(result, 1.12)
        result = pygame.math.clamp(2.5, 1.12, 5.0)
        self.assertAlmostEqual(result, 2.5)
        self.assertRaises(TypeError, pygame.math.clamp, 10)
        self.assertRaises(TypeError, pygame.math.clamp, 'hello', 'py', 'thon')