import re
import xml
import xml.etree.ElementTree as ET
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union
from xml.etree.ElementTree import TreeBuilder
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict
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