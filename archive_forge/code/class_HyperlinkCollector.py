import json
import re
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from queue import PriorityQueue, Queue
from threading import Thread
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, cast
from urllib.parse import unquote, urlparse, urlunparse
from docutils import nodes
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, TooManyRedirects
from sphinx.application import Sphinx
from sphinx.builders.dummy import DummyBuilder
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line
class HyperlinkCollector(SphinxPostTransform):
    builders = ('linkcheck',)
    default_priority = 800

    def run(self, **kwargs: Any) -> None:
        builder = cast(CheckExternalLinksBuilder, self.app.builder)
        hyperlinks = builder.hyperlinks

        def add_uri(uri: str, node: nodes.Element) -> None:
            newuri = self.app.emit_firstresult('linkcheck-process-uri', uri)
            if newuri:
                uri = newuri
            lineno = get_node_line(node)
            uri_info = Hyperlink(uri, self.env.docname, lineno)
            if uri not in hyperlinks:
                hyperlinks[uri] = uri_info
        for refnode in self.document.findall(nodes.reference):
            if 'refuri' not in refnode:
                continue
            uri = refnode['refuri']
            add_uri(uri, refnode)
        for imgnode in self.document.findall(nodes.image):
            uri = imgnode['candidates'].get('?')
            if uri and '://' in uri:
                add_uri(uri, imgnode)
        for rawnode in self.document.findall(nodes.raw):
            uri = rawnode.get('source')
            if uri and '://' in uri:
                add_uri(uri, rawnode)