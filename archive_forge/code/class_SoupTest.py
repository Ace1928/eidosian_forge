import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
class SoupTest(object):

    @property
    def default_builder(self):
        return default_builder

    def soup(self, markup, **kwargs):
        """Build a Beautiful Soup object from markup."""
        builder = kwargs.pop('builder', self.default_builder)
        return BeautifulSoup(markup, builder=builder, **kwargs)

    def document_for(self, markup, **kwargs):
        """Turn an HTML fragment into a document.

        The details depend on the builder.
        """
        return self.default_builder(**kwargs).test_fragment_to_document(markup)

    def assert_soup(self, to_parse, compare_parsed_to=None):
        """Parse some markup using Beautiful Soup and verify that
        the output markup is as expected.
        """
        builder = self.default_builder
        obj = BeautifulSoup(to_parse, builder=builder)
        if compare_parsed_to is None:
            compare_parsed_to = to_parse
        assert obj.decode() == self.document_for(compare_parsed_to)
        assert all((v == 0 for v in list(obj.open_tag_counter.values())))
        assert [obj.ROOT_TAG_NAME] == [x.name for x in obj.tagStack]
    assertSoupEquals = assert_soup

    def assertConnectedness(self, element):
        """Ensure that next_element and previous_element are properly
        set for all descendants of the given element.
        """
        earlier = None
        for e in element.descendants:
            if earlier:
                assert e == earlier.next_element
                assert earlier == e.previous_element
            earlier = e

    def linkage_validator(self, el, _recursive_call=False):
        """Ensure proper linkage throughout the document."""
        descendant = None
        if el.parent is None:
            assert el.previous_element is None, 'Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}'.format(el, el.previous_element, None)
            assert el.previous_sibling is None, 'Bad previous_sibling\nNODE: {}\nPREV: {}\nEXPECTED: {}'.format(el, el.previous_sibling, None)
            assert el.next_sibling is None, 'Bad next_sibling\nNODE: {}\nNEXT: {}\nEXPECTED: {}'.format(el, el.next_sibling, None)
        idx = 0
        child = None
        last_child = None
        last_idx = len(el.contents) - 1
        for child in el.contents:
            descendant = None
            if idx == 0:
                if el.parent is not None:
                    assert el.next_element is child, 'Bad next_element\nNODE: {}\nNEXT: {}\nEXPECTED: {}'.format(el, el.next_element, child)
                    assert child.previous_element is el, 'Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}'.format(child, child.previous_element, el)
                    assert child.previous_sibling is None, 'Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED: {}'.format(child, child.previous_sibling, None)
            else:
                assert child.previous_sibling is el.contents[idx - 1], 'Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED {}'.format(child, child.previous_sibling, el.contents[idx - 1])
                assert el.contents[idx - 1].next_sibling is child, 'Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(el.contents[idx - 1], el.contents[idx - 1].next_sibling, child)
                if last_child is not None:
                    assert child.previous_element is last_child, 'Bad previous_element\nNODE: {}\nPREV {}\nEXPECTED {}\nCONTENTS {}'.format(child, child.previous_element, last_child, child.parent.contents)
                    assert last_child.next_element is child, 'Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(last_child, last_child.next_element, child)
            if isinstance(child, Tag) and child.contents:
                descendant = self.linkage_validator(child, True)
                assert descendant.next_sibling is None, 'Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(descendant, descendant.next_sibling, None)
            if descendant is not None:
                last_child = descendant
            else:
                last_child = child
            if idx == last_idx:
                assert child.next_sibling is None, 'Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(child, child.next_sibling, None)
            idx += 1
        child = descendant if descendant is not None else child
        if child is None:
            child = el
        if not _recursive_call and child is not None:
            target = el
            while True:
                if target is None:
                    assert child.next_element is None, 'Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(child, child.next_element, None)
                    break
                elif target.next_sibling is not None:
                    assert child.next_element is target.next_sibling, 'Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(child, child.next_element, target.next_sibling)
                    break
                target = target.parent
            return None
        else:
            return child

    def assert_selects(self, tags, should_match):
        """Make sure that the given tags have the correct text.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
        assert [tag.string for tag in tags] == should_match

    def assert_selects_ids(self, tags, should_match):
        """Make sure that the given tags have the correct IDs.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
        assert [tag['id'] for tag in tags] == should_match