from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
class TestGetNodes:

    def test_get_child(self):
        steven: TreeNode = TreeNode()
        sue = TreeNode(children={'Steven': steven})
        mary = TreeNode(children={'Sue': sue})
        john = TreeNode(children={'Mary': mary})
        assert john._get_item('Mary') is mary
        assert mary._get_item('Sue') is sue
        with pytest.raises(KeyError):
            john._get_item('Kate')
        assert john._get_item('Mary/Sue') is sue
        assert john._get_item('Mary/Sue/Steven') is steven
        assert mary._get_item('Sue/Steven') is steven

    def test_get_upwards(self):
        sue: TreeNode = TreeNode()
        kate: TreeNode = TreeNode()
        mary = TreeNode(children={'Sue': sue, 'Kate': kate})
        john = TreeNode(children={'Mary': mary})
        assert sue._get_item('../') is mary
        assert sue._get_item('../../') is john
        assert sue._get_item('../Kate') is kate

    def test_get_from_root(self):
        sue: TreeNode = TreeNode()
        mary = TreeNode(children={'Sue': sue})
        john = TreeNode(children={'Mary': mary})
        assert sue._get_item('/Mary') is mary