import unittest
from zope.interface.tests import OptimizationTestMixin
class CustomAdapterRegistry(BaseAdapterRegistry):
    _mappingType = self._getMappingType()
    _sequenceType = self._getMutableListType()
    _leafSequenceType = self._getLeafSequenceType()
    _providedType = self._getProvidedType()

    def _addValueToLeaf(self, existing_leaf_sequence, new_item):
        if not existing_leaf_sequence:
            existing_leaf_sequence = self._leafSequenceType()
        existing_leaf_sequence.append(new_item)
        return existing_leaf_sequence

    def _removeValueFromLeaf(self, existing_leaf_sequence, to_remove):
        without_removed = BaseAdapterRegistry._removeValueFromLeaf(self, existing_leaf_sequence, to_remove)
        existing_leaf_sequence[:] = without_removed
        assert to_remove not in existing_leaf_sequence
        return existing_leaf_sequence