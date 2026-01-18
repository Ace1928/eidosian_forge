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
class HyperlinkAvailabilityChecker:

    def __init__(self, env: BuildEnvironment, config: Config) -> None:
        self.config = config
        self.env = env
        self.rate_limits: Dict[str, RateLimit] = {}
        self.rqueue: Queue[CheckResult] = Queue()
        self.workers: List[Thread] = []
        self.wqueue: PriorityQueue[CheckRequest] = PriorityQueue()
        self.to_ignore = [re.compile(x) for x in self.config.linkcheck_ignore]

    def invoke_threads(self) -> None:
        for _i in range(self.config.linkcheck_workers):
            thread = HyperlinkAvailabilityCheckWorker(self.env, self.config, self.rqueue, self.wqueue, self.rate_limits)
            thread.start()
            self.workers.append(thread)

    def shutdown_threads(self) -> None:
        self.wqueue.join()
        for _worker in self.workers:
            self.wqueue.put(CheckRequest(CHECK_IMMEDIATELY, None), False)

    def check(self, hyperlinks: Dict[str, Hyperlink]) -> Generator[CheckResult, None, None]:
        self.invoke_threads()
        total_links = 0
        for hyperlink in hyperlinks.values():
            if self.is_ignored_uri(hyperlink.uri):
                yield CheckResult(hyperlink.uri, hyperlink.docname, hyperlink.lineno, 'ignored', '', 0)
            else:
                self.wqueue.put(CheckRequest(CHECK_IMMEDIATELY, hyperlink), False)
                total_links += 1
        done = 0
        while done < total_links:
            yield self.rqueue.get()
            done += 1
        self.shutdown_threads()

    def is_ignored_uri(self, uri: str) -> bool:
        return any((pat.match(uri) for pat in self.to_ignore))