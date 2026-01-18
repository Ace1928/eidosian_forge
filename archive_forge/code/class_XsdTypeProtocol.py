from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdTypeProtocol(XsdComponentProtocol, Protocol):

    def is_simple(self) -> bool:
        """Returns `True` if it's a simpleType instance, `False` if it's a complexType."""
        ...

    def is_empty(self) -> bool:
        """
        Returns `True` if it's a simpleType instance or a complexType with empty content,
        `False` otherwise.
        """
        ...

    def has_simple_content(self) -> bool:
        """
        Returns `True` if it's a simpleType instance or a complexType with simple content,
        `False` otherwise.
        """
        ...

    def has_mixed_content(self) -> bool:
        """
        Returns `True` if it's a complexType with mixed content, `False` otherwise.
        """
        ...

    def is_element_only(self) -> bool:
        """
        Returns `True` if it's a complexType with element-only content, `False` otherwise.
        """
        ...

    def is_key(self) -> bool:
        """Returns `True` if it's a simpleType derived from xs:ID, `False` otherwise."""
        ...

    def is_qname(self) -> bool:
        """Returns `True` if it's a simpleType derived from xs:QName, `False` otherwise."""
        ...

    def is_notation(self) -> bool:
        """Returns `True` if it's a simpleType derived from xs:NOTATION, `False` otherwise."""
        ...

    def is_valid(self, obj: Any, *args: Any, **kwargs: Any) -> bool:
        """
        Validates an XML object node using the XSD type. The argument *obj* is an element
        for complex type nodes or a text value for simple type nodes. Returns `True` if
        the argument is valid, `False` otherwise.
        """
        ...

    def validate(self, obj: Any, *args: Any, **kwargs: Any) -> None:
        """
        Validates an XML object node using the XSD type. The argument *obj* is an element
        for complex type nodes or a text value for simple type nodes. Raises a `ValueError`
        compatible exception (a `ValueError` or a subclass of it) if the argument is not valid.
        """
        ...

    def decode(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Decodes an XML object node using the XSD type. The argument *obj* is an element
        for complex type nodes or a text value for simple type nodes. Raises a `ValueError`
        or a `TypeError` compatible exception if the argument it's not valid.
        """
        ...

    @property
    def root_type(self) -> 'XsdTypeProtocol':
        """
        The type at base of the definition of the XSD type. For a special type is the type
        itself. For an atomic type is the primitive type. For a list is the primitive type
        of the item. For a union is the base union type. For a complex type is xs:anyType.
        """
        ...