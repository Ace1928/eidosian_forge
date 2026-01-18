from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
class IElement(Interface):
    """
    Interface to XML element nodes.

    See L{Element} for a detailed example of its general use.

    Warning: this Interface is not yet complete!
    """
    uri = Attribute(" Element's namespace URI ")
    name = Attribute(" Element's local name ")
    defaultUri = Attribute(' Default namespace URI of child elements ')
    attributes = Attribute(' Dictionary of element attributes ')
    children = Attribute(' List of child nodes ')
    parent = Attribute(" Reference to element's parent element ")
    localPrefixes = Attribute(' Dictionary of local prefixes ')

    def toXml(prefixes=None, closeElement=1, defaultUri='', prefixesInScope=None):
        """Serializes object to a (partial) XML document

        @param prefixes: dictionary that maps namespace URIs to suggested
                         prefix names.
        @type prefixes: L{dict}

        @param closeElement: flag that determines whether to include the
            closing tag of the element in the serialized string. A value of
            C{0} only generates the element's start tag. A value of C{1} yields
            a complete serialization.
        @type closeElement: L{int}

        @param defaultUri: Initial default namespace URI. This is most useful
            for partial rendering, where the logical parent element (of which
            the starttag was already serialized) declares a default namespace
            that should be inherited.
        @type defaultUri: L{str}

        @param prefixesInScope: list of prefixes that are assumed to be
            declared by ancestors.
        @type prefixesInScope: L{list}

        @return: (partial) serialized XML
        @rtype: L{str}
        """

    def addElement(name, defaultUri=None, content=None):
        """
        Create an element and add as child.

        The new element is added to this element as a child, and will have
        this element as its parent.

        @param name: element name. This can be either a L{str} object that
            contains the local name, or a tuple of (uri, local_name) for a
            fully qualified name. In the former case, the namespace URI is
            inherited from this element.
        @type name: L{str} or L{tuple} of (L{str}, L{str})

        @param defaultUri: default namespace URI for child elements. If
            L{None}, this is inherited from this element.
        @type defaultUri: L{str}

        @param content: text contained by the new element.
        @type content: L{str}

        @return: the created element
        @rtype: object providing L{IElement}
        """

    def addChild(node):
        """
        Adds a node as child of this element.

        The C{node} will be added to the list of childs of this element, and
        will have this element set as its parent when C{node} provides
        L{IElement}. If C{node} is a L{str} and the current last child is
        character data (L{str}), the text from C{node} is appended to the
        existing last child.

        @param node: the child node.
        @type node: L{str} or object implementing L{IElement}
        """

    def addContent(text):
        """
        Adds character data to this element.

        If the current last child of this element is a string, the text will
        be appended to that string. Otherwise, the text will be added as a new
        child.

        @param text: The character data to be added to this element.
        @type text: L{str}
        """