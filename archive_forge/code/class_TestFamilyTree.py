from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
class TestFamilyTree:

    def test_lonely(self):
        root: TreeNode = TreeNode()
        assert root.parent is None
        assert root.children == {}

    def test_parenting(self):
        john: TreeNode = TreeNode()
        mary: TreeNode = TreeNode()
        mary._set_parent(john, 'Mary')
        assert mary.parent == john
        assert john.children['Mary'] is mary

    def test_no_time_traveller_loops(self):
        john: TreeNode = TreeNode()
        with pytest.raises(InvalidTreeError, match='cannot be a parent of itself'):
            john._set_parent(john, 'John')
        with pytest.raises(InvalidTreeError, match='cannot be a parent of itself'):
            john.children = {'John': john}
        mary: TreeNode = TreeNode()
        rose: TreeNode = TreeNode()
        mary._set_parent(john, 'Mary')
        rose._set_parent(mary, 'Rose')
        with pytest.raises(InvalidTreeError, match='is already a descendant'):
            john._set_parent(rose, 'John')
        with pytest.raises(InvalidTreeError, match='is already a descendant'):
            rose.children = {'John': john}

    def test_parent_swap(self):
        john: TreeNode = TreeNode()
        mary: TreeNode = TreeNode()
        mary._set_parent(john, 'Mary')
        steve: TreeNode = TreeNode()
        mary._set_parent(steve, 'Mary')
        assert mary.parent == steve
        assert steve.children['Mary'] is mary
        assert 'Mary' not in john.children

    def test_multi_child_family(self):
        mary: TreeNode = TreeNode()
        kate: TreeNode = TreeNode()
        john: TreeNode = TreeNode(children={'Mary': mary, 'Kate': kate})
        assert john.children['Mary'] is mary
        assert john.children['Kate'] is kate
        assert mary.parent is john
        assert kate.parent is john

    def test_disown_child(self):
        mary: TreeNode = TreeNode()
        john: TreeNode = TreeNode(children={'Mary': mary})
        mary.orphan()
        assert mary.parent is None
        assert 'Mary' not in john.children

    def test_doppelganger_child(self):
        kate: TreeNode = TreeNode()
        john: TreeNode = TreeNode()
        with pytest.raises(TypeError):
            john.children = {'Kate': 666}
        with pytest.raises(InvalidTreeError, match='Cannot add same node'):
            john.children = {'Kate': kate, 'Evil_Kate': kate}
        john = TreeNode(children={'Kate': kate})
        evil_kate: TreeNode = TreeNode()
        evil_kate._set_parent(john, 'Kate')
        assert john.children['Kate'] is evil_kate

    def test_sibling_relationships(self):
        mary: TreeNode = TreeNode()
        kate: TreeNode = TreeNode()
        ashley: TreeNode = TreeNode()
        TreeNode(children={'Mary': mary, 'Kate': kate, 'Ashley': ashley})
        assert kate.siblings['Mary'] is mary
        assert kate.siblings['Ashley'] is ashley
        assert 'Kate' not in kate.siblings

    def test_ancestors(self):
        tony: TreeNode = TreeNode()
        michael: TreeNode = TreeNode(children={'Tony': tony})
        vito = TreeNode(children={'Michael': michael})
        assert tony.root is vito
        assert tony.parents == (michael, vito)
        assert tony.ancestors == (vito, michael, tony)