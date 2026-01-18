from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
class TestResource:

    def test_from_contents_from_json_schema(self):
        schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
        resource = Resource.from_contents(schema)
        assert resource == Resource(contents=schema, specification=DRAFT202012)

    def test_from_contents_with_no_discernible_information(self):
        """
        Creating a resource with no discernible way to see what
        specification it belongs to (e.g. no ``$schema`` keyword for JSON
        Schema) raises an error.
        """
        with pytest.raises(exceptions.CannotDetermineSpecification):
            Resource.from_contents({'foo': 'bar'})

    def test_from_contents_with_no_discernible_information_and_default(self):
        resource = Resource.from_contents({'foo': 'bar'}, default_specification=Specification.OPAQUE)
        assert resource == Resource.opaque(contents={'foo': 'bar'})

    def test_from_contents_unneeded_default(self):
        schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
        resource = Resource.from_contents(schema, default_specification=Specification.OPAQUE)
        assert resource == Resource(contents=schema, specification=DRAFT202012)

    def test_non_mapping_from_contents(self):
        resource = Resource.from_contents(True, default_specification=ID_AND_CHILDREN)
        assert resource == Resource(contents=True, specification=ID_AND_CHILDREN)

    def test_from_contents_with_fallback(self):
        resource = Resource.from_contents({'foo': 'bar'}, default_specification=Specification.OPAQUE)
        assert resource == Resource.opaque(contents={'foo': 'bar'})

    def test_id_delegates_to_specification(self):
        specification = Specification(name='', id_of=lambda contents: 'urn:fixedID', subresources_of=lambda contents: [], anchors_in=lambda specification, contents: [], maybe_in_subresource=lambda segments, resolver, subresource: resolver)
        resource = Resource(contents={'foo': 'baz'}, specification=specification)
        assert resource.id() == 'urn:fixedID'

    def test_id_strips_empty_fragment(self):
        uri = 'http://example.com/'
        root = ID_AND_CHILDREN.create_resource({'ID': uri + '#'})
        assert root.id() == uri

    def test_subresources_delegates_to_specification(self):
        resource = ID_AND_CHILDREN.create_resource({'children': [{}, 12]})
        assert list(resource.subresources()) == [ID_AND_CHILDREN.create_resource(each) for each in [{}, 12]]

    def test_subresource_with_different_specification(self):
        schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
        resource = ID_AND_CHILDREN.create_resource({'children': [schema]})
        assert list(resource.subresources()) == [DRAFT202012.create_resource(schema)]

    def test_anchors_delegates_to_specification(self):
        resource = ID_AND_CHILDREN.create_resource({'anchors': {'foo': {}, 'bar': 1, 'baz': ''}})
        assert list(resource.anchors()) == [Anchor(name='foo', resource=ID_AND_CHILDREN.create_resource({})), Anchor(name='bar', resource=ID_AND_CHILDREN.create_resource(1)), Anchor(name='baz', resource=ID_AND_CHILDREN.create_resource(''))]

    def test_pointer_to_mapping(self):
        resource = Resource.opaque(contents={'foo': 'baz'})
        resolver = Registry().resolver()
        assert resource.pointer('/foo', resolver=resolver).contents == 'baz'

    def test_pointer_to_array(self):
        resource = Resource.opaque(contents={'foo': {'bar': [3]}})
        resolver = Registry().resolver()
        assert resource.pointer('/foo/bar/0', resolver=resolver).contents == 3

    def test_opaque(self):
        contents = {'foo': 'bar'}
        assert Resource.opaque(contents) == Resource(contents=contents, specification=Specification.OPAQUE)