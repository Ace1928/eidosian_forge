import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.util import split_into
from sphinx.search.en import SearchEnglish
class WordCollector(nodes.NodeVisitor):
    """
    A special visitor that collects words for the `IndexBuilder`.
    """

    def __init__(self, document: nodes.document, lang: SearchLanguage) -> None:
        super().__init__(document)
        self.found_words: List[str] = []
        self.found_titles: List[Tuple[str, str]] = []
        self.found_title_words: List[str] = []
        self.lang = lang

    def is_meta_keywords(self, node: Element) -> bool:
        if isinstance(node, (addnodes.meta, addnodes.docutils_meta)) and node.get('name') == 'keywords':
            meta_lang = node.get('lang')
            if meta_lang is None:
                return True
            elif meta_lang == self.lang.lang:
                return True
        return False

    def dispatch_visit(self, node: Node) -> None:
        if isinstance(node, nodes.comment):
            raise nodes.SkipNode
        elif isinstance(node, nodes.raw):
            if 'html' in node.get('format', '').split():
                nodetext = re.sub('(?is)<style.*?</style>', '', node.astext())
                nodetext = re.sub('(?is)<script.*?</script>', '', nodetext)
                nodetext = re.sub('<[^<]+?>', '', nodetext)
                self.found_words.extend(self.lang.split(nodetext))
            raise nodes.SkipNode
        elif isinstance(node, nodes.Text):
            self.found_words.extend(self.lang.split(node.astext()))
        elif isinstance(node, nodes.title):
            title = node.astext()
            ids = node.parent['ids']
            self.found_titles.append((title, ids[0] if ids else None))
            self.found_title_words.extend(self.lang.split(title))
        elif isinstance(node, Element) and self.is_meta_keywords(node):
            keywords = node['content']
            keywords = [keyword.strip() for keyword in keywords.split(',')]
            self.found_words.extend(keywords)