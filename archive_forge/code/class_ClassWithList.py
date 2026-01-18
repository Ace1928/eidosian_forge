import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
class ClassWithList(HasTraits):
    values = List()
    not_a_trait_list = Instance(CustomList)
    number = Int()
    custom_trait_list = Instance(CustomTraitList)