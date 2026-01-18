import unittest
from collections import Counter
from low_index import *
class TestSimsNode(unittest.TestCase):

    def test_doc(self):
        self.assertIn('A non-abstract SimsNode', SimsNode.__doc__)
        self.assertIn('Create SimsNode for a covering graph ', SimsNode.__init__.__doc__)
        self.assertIn('Copy a SimsNode', SimsNode.__init__.__doc__)

    def test_add_edge(self):
        node = SimsNode(8, 4)
        self.assertEqual(node.rank, 8)
        node.add_edge(2, 1, 1)
        node.add_edge(3, 1, 2)
        node.add_edge(4, 2, 3)
        self.assertTrue(node.verified_add_edge(2, 1, 2))
        self.assertTrue(node.verified_add_edge(2, 2, 3))
        self.assertFalse(node.verified_add_edge(2, 3, 1))
        self.assertFalse(node.is_complete())
        self.assertEqual(node.degree, 3)
        self.assertIn('1--( 3)->2', str(node))
        with self.assertRaises(ValueError):
            node.permutation_rep()

    def test_permutation_rep(self):
        node = SimsNode(2, 2)
        node.add_edge(1, 1, 1)
        node.add_edge(2, 1, 2)
        node.add_edge(2, 2, 1)
        node.add_edge(1, 2, 2)
        self.assertTrue(node.is_complete())
        self.assertEqual(node.degree, 2)
        self.assertEqual(node.permutation_rep(), [[0, 1], [1, 0]])