from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzXml(object):
    """
    Wrapper class for struct `fz_xml`.
    XML document model
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_debug_xml(self, level):
        """
        Class-aware wrapper for `::fz_debug_xml()`.
        	Pretty-print an XML tree to stdout. (Deprecated, use
        	fz_output_xml in preference).
        """
        return _mupdf.FzXml_fz_debug_xml(self, level)

    def fz_detach_xml(self):
        """
        Class-aware wrapper for `::fz_detach_xml()`.
        	Detach a node from the tree, unlinking it from its parent,
        	and setting the document root to the node.
        """
        return _mupdf.FzXml_fz_detach_xml(self)

    def fz_dom_add_attribute(self, att, value):
        """
        Class-aware wrapper for `::fz_dom_add_attribute()`.
        	Add an attribute to an element.

        	Ownership of att and value remain with the caller.
        """
        return _mupdf.FzXml_fz_dom_add_attribute(self, att, value)

    def fz_dom_append_child(self, child):
        """
        Class-aware wrapper for `::fz_dom_append_child()`.
        	Insert an element as the last child of a parent, unlinking the
        	child from its current position if required.
        """
        return _mupdf.FzXml_fz_dom_append_child(self, child)

    def fz_dom_attribute(self, att):
        """
        Class-aware wrapper for `::fz_dom_attribute()`.
        	Retrieve the value of a given attribute from a given element.

        	Returns a borrowed pointer to the value or NULL if not found.
        """
        return _mupdf.FzXml_fz_dom_attribute(self, att)

    def fz_dom_body(self):
        """
        Class-aware wrapper for `::fz_dom_body()`.
        	Return a borrowed reference for the 'body' element of
        	the given DOM.
        """
        return _mupdf.FzXml_fz_dom_body(self)

    def fz_dom_clone(self):
        """
        Class-aware wrapper for `::fz_dom_clone()`.
        	Clone an element (and its children).

        	A borrowed reference to the clone is returned. The clone is not
        	yet linked into the DOM.
        """
        return _mupdf.FzXml_fz_dom_clone(self)

    def fz_dom_create_element(self, tag):
        """
        Class-aware wrapper for `::fz_dom_create_element()`.
        	Create an element of a given tag type for the given DOM.

        	The element is not linked into the DOM yet.
        """
        return _mupdf.FzXml_fz_dom_create_element(self, tag)

    def fz_dom_create_text_node(self, text):
        """
        Class-aware wrapper for `::fz_dom_create_text_node()`.
        	Create a text node for the given DOM.

        	The element is not linked into the DOM yet.
        """
        return _mupdf.FzXml_fz_dom_create_text_node(self, text)

    def fz_dom_document_element(self):
        """
        Class-aware wrapper for `::fz_dom_document_element()`.
        	Return a borrowed reference for the document (the top
        	level element) of the DOM.
        """
        return _mupdf.FzXml_fz_dom_document_element(self)

    def fz_dom_find(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_dom_find()`.
        	Find the first element matching the requirements in a depth first traversal from elt.

        	The tagname must match tag, unless tag is NULL, when all tag names are considered to match.

        	If att is NULL, then all tags match.
        	Otherwise:
        		If match is NULL, then only nodes that have an att attribute match.
        		If match is non-NULL, then only nodes that have an att attribute that matches match match.

        	Returns NULL (if no match found), or a borrowed reference to the first matching element.
        """
        return _mupdf.FzXml_fz_dom_find(self, tag, att, match)

    def fz_dom_find_next(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_dom_find_next()`.
        	Find the next element matching the requirements.
        """
        return _mupdf.FzXml_fz_dom_find_next(self, tag, att, match)

    def fz_dom_first_child(self):
        """
        Class-aware wrapper for `::fz_dom_first_child()`.
        	Return a borrowed reference to the first child of a node,
        	or NULL if there isn't one.
        """
        return _mupdf.FzXml_fz_dom_first_child(self)

    def fz_dom_get_attribute(self, i, att):
        """
        Class-aware wrapper for `::fz_dom_get_attribute()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_dom_get_attribute(int i)` => `(const char *, const char *att)`

        	Enumerate through the attributes of an element.

        	Call with i=0,1,2,3... to enumerate attributes.

        	On return *att and the return value will be NULL if there are not
        	that many attributes to read. Otherwise, *att will be filled in
        	with a borrowed pointer to the attribute name, and the return
        	value will be a borrowed pointer to the value.
        """
        return _mupdf.FzXml_fz_dom_get_attribute(self, i, att)

    def fz_dom_insert_after(self, new_elt):
        """
        Class-aware wrapper for `::fz_dom_insert_after()`.
        	Insert an element (new_elt), after another element (node),
        	unlinking the new_elt from its current position if required.
        """
        return _mupdf.FzXml_fz_dom_insert_after(self, new_elt)

    def fz_dom_insert_before(self, new_elt):
        """
        Class-aware wrapper for `::fz_dom_insert_before()`.
        	Insert an element (new_elt), before another element (node),
        	unlinking the new_elt from its current position if required.
        """
        return _mupdf.FzXml_fz_dom_insert_before(self, new_elt)

    def fz_dom_next(self):
        """
        Class-aware wrapper for `::fz_dom_next()`.
        	Return a borrowed reference to the next sibling of a node,
        	or NULL if there isn't one.
        """
        return _mupdf.FzXml_fz_dom_next(self)

    def fz_dom_parent(self):
        """
        Class-aware wrapper for `::fz_dom_parent()`.
        	Return a borrowed reference to the parent of a node,
        	or NULL if there isn't one.
        """
        return _mupdf.FzXml_fz_dom_parent(self)

    def fz_dom_previous(self):
        """
        Class-aware wrapper for `::fz_dom_previous()`.
        	Return a borrowed reference to the previous sibling of a node,
        	or NULL if there isn't one.
        """
        return _mupdf.FzXml_fz_dom_previous(self)

    def fz_dom_remove(self):
        """
        Class-aware wrapper for `::fz_dom_remove()`.
        	Remove an element from the DOM. The element can be added back elsewhere
        	if required.

        	No reference counting changes for the element.
        """
        return _mupdf.FzXml_fz_dom_remove(self)

    def fz_dom_remove_attribute(self, att):
        """
        Class-aware wrapper for `::fz_dom_remove_attribute()`.
        	Remove an attribute from an element.
        """
        return _mupdf.FzXml_fz_dom_remove_attribute(self, att)

    def fz_new_display_list_from_svg_xml(self, xml, base_uri, dir, w, h):
        """
        Class-aware wrapper for `::fz_new_display_list_from_svg_xml()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_new_display_list_from_svg_xml(::fz_xml *xml, const char *base_uri, ::fz_archive *dir)` => `(fz_display_list *, float w, float h)`

        	Parse an SVG document into a display-list.
        """
        return _mupdf.FzXml_fz_new_display_list_from_svg_xml(self, xml, base_uri, dir, w, h)

    def fz_new_image_from_svg_xml(self, xml, base_uri, dir):
        """
        Class-aware wrapper for `::fz_new_image_from_svg_xml()`.
        	Create a scalable image from an SVG document.
        """
        return _mupdf.FzXml_fz_new_image_from_svg_xml(self, xml, base_uri, dir)

    def fz_xml_att(self, att):
        """
        Class-aware wrapper for `::fz_xml_att()`.
        	Return the value of an attribute of an XML node.
        	NULL if the attribute doesn't exist.
        """
        return _mupdf.FzXml_fz_xml_att(self, att)

    def fz_xml_att_alt(self, one, two):
        """
        Class-aware wrapper for `::fz_xml_att_alt()`.
        	Return the value of an attribute of an XML node.
        	If the first attribute doesn't exist, try the second.
        	NULL if neither attribute exists.
        """
        return _mupdf.FzXml_fz_xml_att_alt(self, one, two)

    def fz_xml_att_eq(self, name, match):
        """
        Class-aware wrapper for `::fz_xml_att_eq()`.
        	Check for a matching attribute on an XML node.

        	If the node has the requested attribute (name), and the value
        	matches (match) then return 1. Otherwise, 0.
        """
        return _mupdf.FzXml_fz_xml_att_eq(self, name, match)

    def fz_xml_down(self):
        """
        Class-aware wrapper for `::fz_xml_down()`.
        	Return first child of XML node.
        """
        return _mupdf.FzXml_fz_xml_down(self)

    def fz_xml_find(self, tag):
        """
        Class-aware wrapper for `::fz_xml_find()`.
        	Search the siblings of XML nodes starting with item looking for
        	the first with the given tag.

        	Return NULL if none found.
        """
        return _mupdf.FzXml_fz_xml_find(self, tag)

    def fz_xml_find_dfs(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_xml_find_dfs()`.
        	Perform a depth first search from item, returning the first
        	child that matches the given tag (or any tag if tag is NULL),
        	with the given attribute (if att is non NULL), that matches
        	match (if match is non NULL).
        """
        return _mupdf.FzXml_fz_xml_find_dfs(self, tag, att, match)

    def fz_xml_find_dfs_top(self, tag, att, match, top):
        """
        Class-aware wrapper for `::fz_xml_find_dfs_top()`.
        	Perform a depth first search from item, returning the first
        	child that matches the given tag (or any tag if tag is NULL),
        	with the given attribute (if att is non NULL), that matches
        	match (if match is non NULL). The search stops if it ever
        	reaches the top of the tree, or the declared 'top' item.
        """
        return _mupdf.FzXml_fz_xml_find_dfs_top(self, tag, att, match, top)

    def fz_xml_find_down(self, tag):
        """
        Class-aware wrapper for `::fz_xml_find_down()`.
        	Search the siblings of XML nodes starting with the first child
        	of item looking for the first with the given tag.

        	Return NULL if none found.
        """
        return _mupdf.FzXml_fz_xml_find_down(self, tag)

    def fz_xml_find_down_match(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_xml_find_down_match()`.
        	Search the siblings of XML nodes starting with the first child
        	of item looking for the first with the given tag (or any tag if
        	tag is NULL), and with a matching attribute.

        	Return NULL if none found.
        """
        return _mupdf.FzXml_fz_xml_find_down_match(self, tag, att, match)

    def fz_xml_find_match(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_xml_find_match()`.
        	Search the siblings of XML nodes starting with item looking for
        	the first with the given tag (or any tag if tag is NULL), and
        	with a matching attribute.

        	Return NULL if none found.
        """
        return _mupdf.FzXml_fz_xml_find_match(self, tag, att, match)

    def fz_xml_find_next(self, tag):
        """
        Class-aware wrapper for `::fz_xml_find_next()`.
        	Search the siblings of XML nodes starting with the first sibling
        	of item looking for the first with the given tag.

        	Return NULL if none found.
        """
        return _mupdf.FzXml_fz_xml_find_next(self, tag)

    def fz_xml_find_next_dfs(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_xml_find_next_dfs()`.
        	Perform a depth first search onwards from item, returning the first
        	child that matches the given tag (or any tag if tag is NULL),
        	with the given attribute (if att is non NULL), that matches
        	match (if match is non NULL).
        """
        return _mupdf.FzXml_fz_xml_find_next_dfs(self, tag, att, match)

    def fz_xml_find_next_dfs_top(self, tag, att, match, top):
        """
        Class-aware wrapper for `::fz_xml_find_next_dfs_top()`.
        	Perform a depth first search onwards from item, returning the first
        	child that matches the given tag (or any tag if tag is NULL),
        	with the given attribute (if att is non NULL), that matches
        	match (if match is non NULL). The search stops if it ever reaches
        	the top of the tree, or the declared 'top' item.
        """
        return _mupdf.FzXml_fz_xml_find_next_dfs_top(self, tag, att, match, top)

    def fz_xml_find_next_match(self, tag, att, match):
        """
        Class-aware wrapper for `::fz_xml_find_next_match()`.
        	Search the siblings of XML nodes starting with the first sibling
        	of item looking for the first with the given tag (or any tag if tag
        	is NULL), and with a matching attribute.

        	Return NULL if none found.
        """
        return _mupdf.FzXml_fz_xml_find_next_match(self, tag, att, match)

    def fz_xml_is_tag(self, name):
        """
        Class-aware wrapper for `::fz_xml_is_tag()`.
        	Return true if the tag name matches.
        """
        return _mupdf.FzXml_fz_xml_is_tag(self, name)

    def fz_xml_next(self):
        """
        Class-aware wrapper for `::fz_xml_next()`.
        	Return next sibling of XML node.
        """
        return _mupdf.FzXml_fz_xml_next(self)

    def fz_xml_prev(self):
        """
        Class-aware wrapper for `::fz_xml_prev()`.
        	Return previous sibling of XML node.
        """
        return _mupdf.FzXml_fz_xml_prev(self)

    def fz_xml_root(self):
        """
        Class-aware wrapper for `::fz_xml_root()`.
        	Return the topmost XML node of a document.
        """
        return _mupdf.FzXml_fz_xml_root(self)

    def fz_xml_tag(self):
        """
        Class-aware wrapper for `::fz_xml_tag()`.
        	Return tag of XML node. Return NULL for text nodes.
        """
        return _mupdf.FzXml_fz_xml_tag(self)

    def fz_xml_text(self):
        """
        Class-aware wrapper for `::fz_xml_text()`.
        	Return the text content of an XML node.
        	Return NULL if the node is a tag.
        """
        return _mupdf.FzXml_fz_xml_text(self)

    def fz_xml_up(self):
        """
        Class-aware wrapper for `::fz_xml_up()`.
        	Return parent of XML node.
        """
        return _mupdf.FzXml_fz_xml_up(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `fz_keep_xml()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_xml`.
        """
        _mupdf.FzXml_swiginit(self, _mupdf.new_FzXml(*args))
    __swig_destroy__ = _mupdf.delete_FzXml

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzXml_m_internal_value(self)
    m_internal = property(_mupdf.FzXml_m_internal_get, _mupdf.FzXml_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzXml_s_num_instances_get, _mupdf.FzXml_s_num_instances_set)