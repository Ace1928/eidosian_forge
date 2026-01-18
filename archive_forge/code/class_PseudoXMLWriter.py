from typing import Any
from docutils.writers.docutils_xml import Writer as BaseXMLWriter
from sphinx.builders import Builder
class PseudoXMLWriter(BaseXMLWriter):
    supported = ('pprint', 'pformat', 'pseudoxml')
    'Formats this writer supports.'
    config_section = 'pseudoxml writer'
    config_section_dependencies = ('writers',)
    output = None
    'Final translated form of `document`.'

    def __init__(self, builder: Builder) -> None:
        super().__init__()
        self.builder = builder

    def translate(self) -> None:
        self.output = self.document.pformat()

    def supports(self, format: str) -> bool:
        """This writer supports all format-specific elements."""
        return True