from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
class TestResolver:

    def test_lookup_exact_uri(self):
        resource = Resource.opaque(contents={'foo': 'baz'})
        resolver = Registry({'http://example.com/1': resource}).resolver()
        resolved = resolver.lookup('http://example.com/1')
        assert resolved.contents == resource.contents

    def test_lookup_subresource(self):
        root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'http://example.com/a', 'foo': 12}]})
        registry = root @ Registry()
        resolved = registry.resolver().lookup('http://example.com/a')
        assert resolved.contents == {'ID': 'http://example.com/a', 'foo': 12}

    def test_lookup_anchor_with_id(self):
        root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'anchors': {'foo': 12}})
        registry = root @ Registry()
        resolved = registry.resolver().lookup('http://example.com/#foo')
        assert resolved.contents == 12

    def test_lookup_anchor_without_id(self):
        root = ID_AND_CHILDREN.create_resource({'anchors': {'foo': 12}})
        resolver = Registry().with_resource('urn:example', root).resolver()
        resolved = resolver.lookup('urn:example#foo')
        assert resolved.contents == 12

    def test_lookup_unknown_reference(self):
        resolver = Registry().resolver()
        ref = 'http://example.com/does/not/exist'
        with pytest.raises(exceptions.Unresolvable) as e:
            resolver.lookup(ref)
        assert e.value == exceptions.Unresolvable(ref=ref)

    def test_lookup_non_existent_pointer(self):
        resource = Resource.opaque({'foo': {}})
        resolver = Registry({'http://example.com/1': resource}).resolver()
        ref = 'http://example.com/1#/foo/bar'
        with pytest.raises(exceptions.Unresolvable) as e:
            resolver.lookup(ref)
        assert e.value == exceptions.PointerToNowhere(ref='/foo/bar', resource=resource)
        assert str(e.value) == "'/foo/bar' does not exist within {'foo': {}}"

    def test_lookup_non_existent_pointer_to_array_index(self):
        resource = Resource.opaque([1, 2, 4, 8])
        resolver = Registry({'http://example.com/1': resource}).resolver()
        ref = 'http://example.com/1#/10'
        with pytest.raises(exceptions.Unresolvable) as e:
            resolver.lookup(ref)
        assert e.value == exceptions.PointerToNowhere(ref='/10', resource=resource)

    def test_lookup_pointer_to_empty_string(self):
        resolver = Registry().resolver_with_root(Resource.opaque({'': {}}))
        assert resolver.lookup('#/').contents == {}

    def test_lookup_non_existent_pointer_to_empty_string(self):
        resource = Resource.opaque({'foo': {}})
        resolver = Registry().resolver_with_root(resource)
        with pytest.raises(exceptions.Unresolvable, match="^'/' does not exist within {'foo': {}}.*'#'") as e:
            resolver.lookup('#/')
        assert e.value == exceptions.PointerToNowhere(ref='/', resource=resource)

    def test_lookup_non_existent_anchor(self):
        root = ID_AND_CHILDREN.create_resource({'anchors': {}})
        resolver = Registry().with_resource('urn:example', root).resolver()
        resolved = resolver.lookup('urn:example')
        assert resolved.contents == root.contents
        ref = 'urn:example#noSuchAnchor'
        with pytest.raises(exceptions.Unresolvable) as e:
            resolver.lookup(ref)
        assert "'noSuchAnchor' does not exist" in str(e.value)
        assert e.value == exceptions.NoSuchAnchor(ref='urn:example', resource=root, anchor='noSuchAnchor')

    def test_lookup_invalid_JSON_pointerish_anchor(self):
        resolver = Registry().resolver_with_root(ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'foo': {'bar': 12}}))
        valid = resolver.lookup('#/foo/bar')
        assert valid.contents == 12
        with pytest.raises(exceptions.InvalidAnchor) as e:
            resolver.lookup('#foo/bar')
        assert " '#/foo/bar'" in str(e.value)

    def test_lookup_retrieved_resource(self):
        resource = Resource.opaque(contents={'foo': 'baz'})
        resolver = Registry(retrieve=lambda uri: resource).resolver()
        resolved = resolver.lookup('http://example.com/')
        assert resolved.contents == resource.contents

    def test_lookup_failed_retrieved_resource(self):
        """
        Unretrievable exceptions are also wrapped in Unresolvable.
        """
        uri = 'http://example.com/'
        registry = Registry(retrieve=blow_up)
        with pytest.raises(exceptions.Unretrievable):
            registry.get_or_retrieve(uri)
        resolver = registry.resolver()
        with pytest.raises(exceptions.Unresolvable):
            resolver.lookup(uri)

    def test_repeated_lookup_from_retrieved_resource(self):
        """
        A (custom-)retrieved resource is added to the registry returned by
        looking it up.
        """
        resource = Resource.opaque(contents={'foo': 'baz'})
        once = [resource]

        def retrieve(uri):
            return once.pop()
        resolver = Registry(retrieve=retrieve).resolver()
        resolved = resolver.lookup('http://example.com/')
        assert resolved.contents == resource.contents
        resolved = resolved.resolver.lookup('http://example.com/')
        assert resolved.contents == resource.contents

    def test_repeated_anchor_lookup_from_retrieved_resource(self):
        resource = Resource.opaque(contents={'foo': 'baz'})
        once = [resource]

        def retrieve(uri):
            return once.pop()
        resolver = Registry(retrieve=retrieve).resolver()
        resolved = resolver.lookup('http://example.com/')
        assert resolved.contents == resource.contents
        resolved = resolved.resolver.lookup('#')
        assert resolved.contents == resource.contents

    def test_in_subresource(self):
        root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'child/', 'children': [{'ID': 'grandchild'}]}]})
        registry = root @ Registry()
        resolver = registry.resolver()
        first = resolver.lookup('http://example.com/')
        assert first.contents == root.contents
        with pytest.raises(exceptions.Unresolvable):
            first.resolver.lookup('grandchild')
        sub = first.resolver.in_subresource(ID_AND_CHILDREN.create_resource(first.contents['children'][0]))
        second = sub.lookup('grandchild')
        assert second.contents == {'ID': 'grandchild'}

    def test_in_pointer_subresource(self):
        root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'child/', 'children': [{'ID': 'grandchild'}]}]})
        registry = root @ Registry()
        resolver = registry.resolver()
        first = resolver.lookup('http://example.com/')
        assert first.contents == root.contents
        with pytest.raises(exceptions.Unresolvable):
            first.resolver.lookup('grandchild')
        second = first.resolver.lookup('#/children/0')
        third = second.resolver.lookup('grandchild')
        assert third.contents == {'ID': 'grandchild'}

    def test_dynamic_scope(self):
        one = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'child/', 'children': [{'ID': 'grandchild'}]}]})
        two = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/two', 'children': [{'ID': 'two-child/'}]})
        registry = [one, two] @ Registry()
        resolver = registry.resolver()
        first = resolver.lookup('http://example.com/')
        second = first.resolver.lookup('#/children/0')
        third = second.resolver.lookup('grandchild')
        fourth = third.resolver.lookup('http://example.com/two')
        assert list(fourth.resolver.dynamic_scope()) == [('http://example.com/child/grandchild', fourth.resolver._registry), ('http://example.com/child/', fourth.resolver._registry), ('http://example.com/', fourth.resolver._registry)]
        assert list(third.resolver.dynamic_scope()) == [('http://example.com/child/', third.resolver._registry), ('http://example.com/', third.resolver._registry)]
        assert list(second.resolver.dynamic_scope()) == [('http://example.com/', second.resolver._registry)]
        assert list(first.resolver.dynamic_scope()) == []