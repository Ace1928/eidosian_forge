import unittest
from zope.interface.tests import OptimizationTestMixin
class CustomTypesBaseAdapterRegistryTests(BaseAdapterRegistryTests):
    """
    This class may be extended by other packages to test their own
    adapter registries that use custom types. (So be cautious about
    breaking changes.)

    One known user is ``zope.component.persistentregistry``.
    """

    def _getMappingType(self):
        return CustomMapping

    def _getProvidedType(self):
        return CustomProvided

    def _getMutableListType(self):
        return CustomSequence

    def _getLeafSequenceType(self):
        return CustomLeafSequence

    def _getBaseAdapterRegistry(self):
        from zope.interface.adapter import BaseAdapterRegistry

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
        return CustomAdapterRegistry

    def assertLeafIdentity(self, leaf1, leaf2):
        self.assertIs(leaf1, leaf2)