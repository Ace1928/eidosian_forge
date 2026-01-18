import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Union
import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _parse_front_matter(self, content: str) -> dict:
    """Parse front matter metadata from the content and return it as a dict."""
    if not self.collect_metadata:
        return {}
    match = self.FRONT_MATTER_REGEX.search(content)
    if not match:
        return {}
    placeholders: Dict[str, str] = {}
    replace_template_var = functools.partial(self._replace_template_var, placeholders)
    front_matter_text = self.TEMPLATE_VARIABLE_REGEX.sub(replace_template_var, match.group(1))
    try:
        front_matter = yaml.safe_load(front_matter_text)
        front_matter = self._restore_template_vars(front_matter, placeholders)
        if 'tags' in front_matter and isinstance(front_matter['tags'], str):
            front_matter['tags'] = front_matter['tags'].split(', ')
        return front_matter
    except yaml.parser.ParserError:
        logger.warning('Encountered non-yaml frontmatter')
        return {}