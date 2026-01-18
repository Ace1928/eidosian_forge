import re
import xml
import xml.etree.ElementTree as ET
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union
from xml.etree.ElementTree import TreeBuilder
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict
class XMLOutputParser(BaseTransformOutputParser):
    """Parse an output using xml format."""
    tags: Optional[List[str]] = None
    encoding_matcher: re.Pattern = re.compile('<([^>]*encoding[^>]*)>\\n(.*)', re.MULTILINE | re.DOTALL)
    parser: Literal['defusedxml', 'xml'] = 'defusedxml'
    "Parser to use for XML parsing. Can be either 'defusedxml' or 'xml'.\n    \n    * 'defusedxml' is the default parser and is used to prevent XML vulnerabilities \n       present in some distributions of Python's standard library xml.\n       `defusedxml` is a wrapper around the standard library parser that\n       sets up the parser with secure defaults.\n    * 'xml' is the standard library parser.\n    \n    Use `xml` only if you are sure that your distribution of the standard library\n    is not vulnerable to XML vulnerabilities. \n    \n    Please review the following resources for more information:\n    \n    * https://docs.python.org/3/library/xml.html#xml-vulnerabilities\n    * https://github.com/tiran/defusedxml \n    \n    The standard library relies on libexpat for parsing XML:\n    https://github.com/libexpat/libexpat \n    "

    def get_format_instructions(self) -> str:
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> Dict[str, Union[str, List[Any]]]:
        if self.parser == 'defusedxml':
            try:
                from defusedxml import ElementTree as DET
            except ImportError:
                raise ImportError('defusedxml is not installed. Please install it to use the defusedxml parser.You can install it with `pip install defusedxml`See https://github.com/tiran/defusedxml for more details')
            _ET = DET
        else:
            _ET = ET
        match = re.search('```(xml)?(.*)```', text, re.DOTALL)
        if match is not None:
            text = match.group(2)
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)
        text = text.strip()
        try:
            root = ET.fromstring(text)
            return self._root_to_dict(root)
        except ET.ParseError as e:
            msg = f'Failed to parse XML format from completion {text}. Got: {e}'
            raise OutputParserException(msg, llm_output=text) from e

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        for chunk in input:
            yield from streaming_parser.parse(chunk)
        streaming_parser.close()

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        async for chunk in input:
            for output in streaming_parser.parse(chunk):
                yield output
        streaming_parser.close()

    def _root_to_dict(self, root: ET.Element) -> Dict[str, Union[str, List[Any]]]:
        """Converts xml tree to python dictionary."""
        if root.text and bool(re.search('\\S', root.text)):
            return {root.tag: root.text}
        result: Dict = {root.tag: []}
        for child in root:
            if len(child) == 0:
                result[root.tag].append({child.tag: child.text})
            else:
                result[root.tag].append(self._root_to_dict(child))
        return result

    @property
    def _type(self) -> str:
        return 'xml'