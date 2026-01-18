import unittest
import pygame.constants
class KConstantsTests(unittest.TestCase):
    """Test K_* (key) constants."""
    K_SPECIFIC_NAMES = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'QUOTE', 'BACKQUOTE', 'EXCLAIM', 'QUOTEDBL', 'HASH', 'DOLLAR', 'AMPERSAND', 'LEFTPAREN', 'RIGHTPAREN', 'ASTERISK', 'PLUS', 'COLON', 'LESS', 'GREATER', 'QUESTION', 'AT', 'CARET', 'UNDERSCORE', 'PERCENT')
    K_NAMES = tuple(('K_' + n for n in K_AND_KSCAN_COMMON_NAMES + K_SPECIFIC_NAMES))

    def test_k__existence(self):
        """Ensures K constants exist."""
        for name in self.K_NAMES:
            self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')

    def test_k__type(self):
        """Ensures K constants are the correct type."""
        for name in self.K_NAMES:
            value = getattr(pygame.constants, name)
            self.assertIs(type(value), int)

    def test_k__value_overlap(self):
        """Ensures no unexpected K constant values overlap."""
        EXPECTED_OVERLAPS = {frozenset(('K_' + n for n in item)) for item in K_AND_KSCAN_COMMON_OVERLAPS}
        overlaps = create_overlap_set(self.K_NAMES)
        self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)