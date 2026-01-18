from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
class TestOpaqueSpecification:
    THINGS = [{'foo': 'bar'}, True, 37, 'foo', object()]

    @pytest.mark.parametrize('thing', THINGS)
    def test_no_id(self, thing):
        """
        An arbitrary thing has no ID.
        """
        assert Specification.OPAQUE.id_of(thing) is None

    @pytest.mark.parametrize('thing', THINGS)
    def test_no_subresources(self, thing):
        """
        An arbitrary thing has no subresources.
        """
        assert list(Specification.OPAQUE.subresources_of(thing)) == []

    @pytest.mark.parametrize('thing', THINGS)
    def test_no_anchors(self, thing):
        """
        An arbitrary thing has no anchors.
        """
        assert list(Specification.OPAQUE.anchors_in(thing)) == []