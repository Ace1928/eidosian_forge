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
class HyperlinkAvailabilityCheckWorker(Thread):
    """A worker class for checking the availability of hyperlinks."""

    def __init__(self, env: BuildEnvironment, config: Config, rqueue: 'Queue[CheckResult]', wqueue: 'Queue[CheckRequest]', rate_limits: Dict[str, RateLimit]) -> None:
        self.config = config
        self.env = env
        self.rate_limits = rate_limits
        self.rqueue = rqueue
        self.wqueue = wqueue
        self.anchors_ignore = [re.compile(x) for x in self.config.linkcheck_anchors_ignore]
        self.documents_exclude = [re.compile(doc) for doc in self.config.linkcheck_exclude_documents]
        self.auth = [(re.compile(pattern), auth_info) for pattern, auth_info in self.config.linkcheck_auth]
        super().__init__(daemon=True)

    def run(self) -> None:
        kwargs = {}
        if self.config.linkcheck_timeout:
            kwargs['timeout'] = self.config.linkcheck_timeout

        def get_request_headers() -> Dict[str, str]:
            url = urlparse(uri)
            candidates = ['%s://%s' % (url.scheme, url.netloc), '%s://%s/' % (url.scheme, url.netloc), uri, '*']
            for u in candidates:
                if u in self.config.linkcheck_request_headers:
                    headers = deepcopy(DEFAULT_REQUEST_HEADERS)
                    headers.update(self.config.linkcheck_request_headers[u])
                    return headers
            return {}

        def check_uri() -> Tuple[str, str, int]:
            if '#' in uri:
                req_url, anchor = uri.split('#', 1)
                for rex in self.anchors_ignore:
                    if rex.match(anchor):
                        anchor = None
                        break
            else:
                req_url = uri
                anchor = None
            try:
                req_url.encode('ascii')
            except UnicodeError:
                req_url = encode_uri(req_url)
            for pattern, auth_info in self.auth:
                if pattern.match(uri):
                    break
            else:
                auth_info = None
            kwargs['headers'] = get_request_headers()
            try:
                if anchor and self.config.linkcheck_anchors:
                    response = requests.get(req_url, stream=True, config=self.config, auth=auth_info, **kwargs)
                    response.raise_for_status()
                    found = check_anchor(response, unquote(anchor))
                    if not found:
                        raise Exception(__("Anchor '%s' not found") % anchor)
                else:
                    try:
                        response = requests.head(req_url, allow_redirects=True, config=self.config, auth=auth_info, **kwargs)
                        response.raise_for_status()
                    except (ConnectionError, HTTPError, TooManyRedirects) as err:
                        if isinstance(err, HTTPError) and err.response.status_code == 429:
                            raise
                        response = requests.get(req_url, stream=True, config=self.config, auth=auth_info, **kwargs)
                        response.raise_for_status()
            except HTTPError as err:
                if err.response.status_code == 401:
                    return ('working', ' - unauthorized', 0)
                elif err.response.status_code == 429:
                    next_check = self.limit_rate(err.response)
                    if next_check is not None:
                        self.wqueue.put(CheckRequest(next_check, hyperlink), False)
                        return ('rate-limited', '', 0)
                    return ('broken', str(err), 0)
                elif err.response.status_code == 503:
                    return ('ignored', str(err), 0)
                else:
                    return ('broken', str(err), 0)
            except Exception as err:
                return ('broken', str(err), 0)
            else:
                netloc = urlparse(req_url).netloc
                try:
                    del self.rate_limits[netloc]
                except KeyError:
                    pass
            if response.url.rstrip('/') == req_url.rstrip('/'):
                return ('working', '', 0)
            else:
                new_url = response.url
                if anchor:
                    new_url += '#' + anchor
                if allowed_redirect(req_url, new_url):
                    return ('working', '', 0)
                elif response.history:
                    code = response.history[-1].status_code
                    return ('redirected', new_url, code)
                else:
                    return ('redirected', new_url, 0)

        def allowed_redirect(url: str, new_url: str) -> bool:
            for from_url, to_url in self.config.linkcheck_allowed_redirects.items():
                if from_url.match(url) and to_url.match(new_url):
                    return True
            return False

        def check(docname: str) -> Tuple[str, str, int]:
            for doc_matcher in self.documents_exclude:
                if doc_matcher.match(docname):
                    info = f'{docname} matched {doc_matcher.pattern} from linkcheck_exclude_documents'
                    return ('ignored', info, 0)
            if len(uri) == 0 or uri.startswith(('#', 'mailto:', 'tel:')):
                return ('unchecked', '', 0)
            elif not uri.startswith(('http:', 'https:')):
                if uri_re.match(uri):
                    return ('unchecked', '', 0)
                else:
                    srcdir = path.dirname(self.env.doc2path(docname))
                    if path.exists(path.join(srcdir, uri)):
                        return ('working', '', 0)
                    else:
                        return ('broken', '', 0)
            for _ in range(self.config.linkcheck_retries):
                status, info, code = check_uri()
                if status != 'broken':
                    break
            return (status, info, code)
        while True:
            check_request = self.wqueue.get()
            try:
                next_check, hyperlink = check_request
                if hyperlink is None:
                    break
                uri, docname, lineno = hyperlink
            except ValueError:
                next_check, uri, docname, lineno = check_request
            if uri is None:
                break
            netloc = urlparse(uri).netloc
            try:
                next_check = self.rate_limits[netloc].next_check
            except KeyError:
                pass
            if next_check > time.time():
                time.sleep(QUEUE_POLL_SECS)
                self.wqueue.put(CheckRequest(next_check, hyperlink), False)
                self.wqueue.task_done()
                continue
            status, info, code = check(docname)
            if status == 'rate-limited':
                logger.info(darkgray('-rate limited-   ') + uri + darkgray(' | sleeping...'))
            else:
                self.rqueue.put(CheckResult(uri, docname, lineno, status, info, code))
            self.wqueue.task_done()

    def limit_rate(self, response: Response) -> Optional[float]:
        next_check = None
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    until = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    pass
                else:
                    next_check = datetime.timestamp(until)
                    delay = (until - datetime.now(timezone.utc)).total_seconds()
            else:
                next_check = time.time() + delay
        netloc = urlparse(response.url).netloc
        if next_check is None:
            max_delay = self.config.linkcheck_rate_limit_timeout
            try:
                rate_limit = self.rate_limits[netloc]
            except KeyError:
                delay = DEFAULT_DELAY
            else:
                last_wait_time = rate_limit.delay
                delay = 2.0 * last_wait_time
                if delay > max_delay and last_wait_time < max_delay:
                    delay = max_delay
            if delay > max_delay:
                return None
            next_check = time.time() + delay
        self.rate_limits[netloc] = RateLimit(delay, next_check)
        return next_check