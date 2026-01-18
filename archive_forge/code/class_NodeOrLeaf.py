from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
class NodeOrLeaf:
    """
    The base class for nodes and leaves.
    """
    __slots__ = ('parent',)
    type: str
    '\n    The type is a string that typically matches the types of the grammar file.\n    '
    parent: 'Optional[BaseNode]'
    '\n    The parent :class:`BaseNode` of this node or leaf.\n    None if this is the root node.\n    '

    def get_root_node(self):
        """
        Returns the root node of a parser tree. The returned node doesn't have
        a parent node like all the other nodes/leaves.
        """
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def get_next_sibling(self):
        """
        Returns the node immediately following this node in this parent's
        children list. If this node does not have a next sibling, it is None
        """
        parent = self.parent
        if parent is None:
            return None
        for i, child in enumerate(parent.children):
            if child is self:
                try:
                    return self.parent.children[i + 1]
                except IndexError:
                    return None

    def get_previous_sibling(self):
        """
        Returns the node immediately preceding this node in this parent's
        children list. If this node does not have a previous sibling, it is
        None.
        """
        parent = self.parent
        if parent is None:
            return None
        for i, child in enumerate(parent.children):
            if child is self:
                if i == 0:
                    return None
                return self.parent.children[i - 1]

    def get_previous_leaf(self):
        """
        Returns the previous leaf in the parser tree.
        Returns `None` if this is the first element in the parser tree.
        """
        if self.parent is None:
            return None
        node = self
        while True:
            c = node.parent.children
            i = c.index(node)
            if i == 0:
                node = node.parent
                if node.parent is None:
                    return None
            else:
                node = c[i - 1]
                break
        while True:
            try:
                node = node.children[-1]
            except AttributeError:
                return node

    def get_next_leaf(self):
        """
        Returns the next leaf in the parser tree.
        Returns None if this is the last element in the parser tree.
        """
        if self.parent is None:
            return None
        node = self
        while True:
            c = node.parent.children
            i = c.index(node)
            if i == len(c) - 1:
                node = node.parent
                if node.parent is None:
                    return None
            else:
                node = c[i + 1]
                break
        while True:
            try:
                node = node.children[0]
            except AttributeError:
                return node

    @abstractproperty
    def start_pos(self) -> Tuple[int, int]:
        """
        Returns the starting position of the prefix as a tuple, e.g. `(3, 4)`.

        :return tuple of int: (line, column)
        """

    @abstractproperty
    def end_pos(self) -> Tuple[int, int]:
        """
        Returns the end position of the prefix as a tuple, e.g. `(3, 4)`.

        :return tuple of int: (line, column)
        """

    @abstractmethod
    def get_start_pos_of_prefix(self):
        """
        Returns the start_pos of the prefix. This means basically it returns
        the end_pos of the last prefix. The `get_start_pos_of_prefix()` of the
        prefix `+` in `2 + 1` would be `(1, 1)`, while the start_pos is
        `(1, 2)`.

        :return tuple of int: (line, column)
        """

    @abstractmethod
    def get_first_leaf(self):
        """
        Returns the first leaf of a node or itself if this is a leaf.
        """

    @abstractmethod
    def get_last_leaf(self):
        """
        Returns the last leaf of a node or itself if this is a leaf.
        """

    @abstractmethod
    def get_code(self, include_prefix=True):
        """
        Returns the code that was the input for the parser for this node.

        :param include_prefix: Removes the prefix (whitespace and comments) of
            e.g. a statement.
        """

    def search_ancestor(self, *node_types: str) -> 'Optional[BaseNode]':
        """
        Recursively looks at the parents of this node or leaf and returns the
        first found node that matches ``node_types``. Returns ``None`` if no
        matching node is found.

        :param node_types: type names that are searched for.
        """
        node = self.parent
        while node is not None:
            if node.type in node_types:
                return node
            node = node.parent
        return None

    def dump(self, *, indent: Optional[Union[int, str]]=4) -> str:
        """
        Returns a formatted dump of the parser tree rooted at this node or leaf. This is
        mainly useful for debugging purposes.

        The ``indent`` parameter is interpreted in a similar way as :py:func:`ast.dump`.
        If ``indent`` is a non-negative integer or string, then the tree will be
        pretty-printed with that indent level. An indent level of 0, negative, or ``""``
        will only insert newlines. ``None`` selects the single line representation.
        Using a positive integer indent indents that many spaces per level. If
        ``indent`` is a string (such as ``"\\t"``), that string is used to indent each
        level.

        :param indent: Indentation style as described above. The default indentation is
            4 spaces, which yields a pretty-printed dump.

        >>> import parso
        >>> print(parso.parse("lambda x, y: x + y").dump())
        Module([
            Lambda([
                Keyword('lambda', (1, 0)),
                Param([
                    Name('x', (1, 7), prefix=' '),
                    Operator(',', (1, 8)),
                ]),
                Param([
                    Name('y', (1, 10), prefix=' '),
                ]),
                Operator(':', (1, 11)),
                PythonNode('arith_expr', [
                    Name('x', (1, 13), prefix=' '),
                    Operator('+', (1, 15), prefix=' '),
                    Name('y', (1, 17), prefix=' '),
                ]),
            ]),
            EndMarker('', (1, 18)),
        ])
        """
        if indent is None:
            newline = False
            indent_string = ''
        elif isinstance(indent, int):
            newline = True
            indent_string = ' ' * indent
        elif isinstance(indent, str):
            newline = True
            indent_string = indent
        else:
            raise TypeError(f"expect 'indent' to be int, str or None, got {indent!r}")

        def _format_dump(node: NodeOrLeaf, indent: str='', top_level: bool=True) -> str:
            result = ''
            node_type = type(node).__name__
            if isinstance(node, Leaf):
                result += f'{indent}{node_type}('
                if isinstance(node, ErrorLeaf):
                    result += f'{node.token_type!r}, '
                elif isinstance(node, TypedLeaf):
                    result += f'{node.type!r}, '
                result += f'{node.value!r}, {node.start_pos!r}'
                if node.prefix:
                    result += f', prefix={node.prefix!r}'
                result += ')'
            elif isinstance(node, BaseNode):
                result += f'{indent}{node_type}('
                if isinstance(node, Node):
                    result += f'{node.type!r}, '
                result += '['
                if newline:
                    result += '\n'
                for child in node.children:
                    result += _format_dump(child, indent=indent + indent_string, top_level=False)
                result += f'{indent}])'
            else:
                raise TypeError(f'unsupported node encountered: {node!r}')
            if not top_level:
                if newline:
                    result += ',\n'
                else:
                    result += ', '
            return result
        return _format_dump(self)