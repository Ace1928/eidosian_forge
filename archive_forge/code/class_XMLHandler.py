from __future__ import annotations
import collections
import dataclasses
import html.parser
import typing
class XMLHandler(html.parser.HTMLParser):
    """
    Handle parsing events emitted by an HTML parser.

    Although this class inherits from the Python's built-in HTMLParser,
    it deliberately favors the lxml API for the sake of speed.
    The HTMLParser API is supported using methods that alias the lxml API methods,
    or transform the input parameters and then call the lxml API methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self.uris: dict[str, list[str]] = {}
        self.node_stack: collections.deque[Node] = collections.deque()
        self.start_methods: dict[tuple[str, str], typing.Callable[[dict[str, str]], None] | None] = {}
        self.end_methods: dict[tuple[str, str], typing.Callable[[], None] | None] = {}
        self.flag_expect_text: bool = False
        self.text: list[str] = []

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        """Handle the start of an XML element."""
        attrs_excluding_xmlns = {}
        namespace_prefixes = set()
        for key, value in attrs.items():
            if key.startswith('xmlns'):
                if value:
                    _, _, declared_prefix = key.partition(':')
                    self.uris.setdefault(declared_prefix, []).append(value)
                    namespace_prefixes.add(declared_prefix)
            else:
                attrs_excluding_xmlns[key] = value
        deployed_prefix, _, name = tag.rpartition(':')
        identifier_list = self.uris.get(deployed_prefix, [uris.get(deployed_prefix, deployed_prefix)])
        if identifier_list:
            identifier = identifier_list[-1]
        else:
            identifier = '= sentinel: no identifier ='
        standard_prefix = prefixes.get(identifier, deployed_prefix)
        node = Node(tag=tag, standard_prefix=standard_prefix, name=name, namespace_prefixes=namespace_prefixes)
        self.node_stack.append(node)
        try:
            start_method = self.start_methods[standard_prefix, name]
        except KeyError:
            if standard_prefix:
                start_method_name = f'start_{standard_prefix}_{name}'
                end_method_name = f'end_{standard_prefix}_{name}'
            else:
                start_method_name = f'start_opml_{name}'
                end_method_name = f'end_opml_{name}'
            start_method = getattr(self, start_method_name, None)
            end_method = getattr(self, end_method_name, None)
            self.start_methods[standard_prefix, name] = start_method
            self.end_methods[standard_prefix, name] = end_method
        if start_method is not None:
            start_method(attrs_excluding_xmlns)

    def end(self, tag: str) -> None:
        """Handle the end of an XML element."""
        while True:
            try:
                node = self.node_stack.pop()
            except IndexError:
                deployed_prefix, _, name = tag.rpartition(':')
                identifier_list = self.uris.get(deployed_prefix, [uris.get(deployed_prefix, deployed_prefix)])
                if identifier_list:
                    identifier = identifier_list[-1]
                else:
                    identifier = '= sentinel: no identifier ='
                standard_prefix = prefixes.get(identifier, deployed_prefix)
                node = Node(tag=tag, standard_prefix=standard_prefix, name=name, namespace_prefixes=set())
            for prefix in node.namespace_prefixes:
                self.uris[prefix].pop()
            end_method = self.end_methods.get((node.standard_prefix, node.name))
            if end_method is not None:
                end_method()
            if node.tag == tag:
                break

    def data(self, data: str) -> None:
        """Handle text content of an element."""
        if self.flag_expect_text:
            self.text.append(data)

    def close(self) -> None:
        """Reset the handler."""
        super().close()
        self.start_methods = {}
        self.end_methods = {}
        self.flag_expect_text = False
        self.text = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle the start of an XML element."""
        return self.start(tag, {key: value or '' for key, value in attrs})
    handle_endtag = end
    handle_data = data