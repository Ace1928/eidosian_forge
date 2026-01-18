import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
class SimpleScrapingLocator(Locator):
    """
    A locator which scrapes HTML pages to locate downloads for a distribution.
    This runs multiple threads to do the I/O; performance is at least as good
    as pip's PackageFinder, which works in an analogous fashion.
    """
    decoders = {'deflate': zlib.decompress, 'gzip': lambda b: gzip.GzipFile(fileobj=BytesIO(b)).read(), 'none': lambda b: b}

    def __init__(self, url, timeout=None, num_workers=10, **kwargs):
        """
        Initialise an instance.
        :param url: The root URL to use for scraping.
        :param timeout: The timeout, in seconds, to be applied to requests.
                        This defaults to ``None`` (no timeout specified).
        :param num_workers: The number of worker threads you want to do I/O,
                            This defaults to 10.
        :param kwargs: Passed to the superclass.
        """
        super(SimpleScrapingLocator, self).__init__(**kwargs)
        self.base_url = ensure_slash(url)
        self.timeout = timeout
        self._page_cache = {}
        self._seen = set()
        self._to_fetch = queue.Queue()
        self._bad_hosts = set()
        self.skip_externals = False
        self.num_workers = num_workers
        self._lock = threading.RLock()
        self._gplock = threading.RLock()
        self.platform_check = False

    def _prepare_threads(self):
        """
        Threads are created only when get_project is called, and terminate
        before it returns. They are there primarily to parallelise I/O (i.e.
        fetching web pages).
        """
        self._threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._fetch)
            t.daemon = True
            t.start()
            self._threads.append(t)

    def _wait_threads(self):
        """
        Tell all the threads to terminate (by sending a sentinel value) and
        wait for them to do so.
        """
        for t in self._threads:
            self._to_fetch.put(None)
        for t in self._threads:
            t.join()
        self._threads = []

    def _get_project(self, name):
        result = {'urls': {}, 'digests': {}}
        with self._gplock:
            self.result = result
            self.project_name = name
            url = urljoin(self.base_url, '%s/' % quote(name))
            self._seen.clear()
            self._page_cache.clear()
            self._prepare_threads()
            try:
                logger.debug('Queueing %s', url)
                self._to_fetch.put(url)
                self._to_fetch.join()
            finally:
                self._wait_threads()
            del self.result
        return result
    platform_dependent = re.compile('\\b(linux_(i\\d86|x86_64|arm\\w+)|win(32|_amd64)|macosx_?\\d+)\\b', re.I)

    def _is_platform_dependent(self, url):
        """
        Does an URL refer to a platform-specific download?
        """
        return self.platform_dependent.search(url)

    def _process_download(self, url):
        """
        See if an URL is a suitable download for a project.

        If it is, register information in the result dictionary (for
        _get_project) about the specific version it's for.

        Note that the return value isn't actually used other than as a boolean
        value.
        """
        if self.platform_check and self._is_platform_dependent(url):
            info = None
        else:
            info = self.convert_url_to_download_info(url, self.project_name)
        logger.debug('process_download: %s -> %s', url, info)
        if info:
            with self._lock:
                self._update_version_data(self.result, info)
        return info

    def _should_queue(self, link, referrer, rel):
        """
        Determine whether a link URL from a referring page and with a
        particular "rel" attribute should be queued for scraping.
        """
        scheme, netloc, path, _, _, _ = urlparse(link)
        if path.endswith(self.source_extensions + self.binary_extensions + self.excluded_extensions):
            result = False
        elif self.skip_externals and (not link.startswith(self.base_url)):
            result = False
        elif not referrer.startswith(self.base_url):
            result = False
        elif rel not in ('homepage', 'download'):
            result = False
        elif scheme not in ('http', 'https', 'ftp'):
            result = False
        elif self._is_platform_dependent(link):
            result = False
        else:
            host = netloc.split(':', 1)[0]
            if host.lower() == 'localhost':
                result = False
            else:
                result = True
        logger.debug('should_queue: %s (%s) from %s -> %s', link, rel, referrer, result)
        return result

    def _fetch(self):
        """
        Get a URL to fetch from the work queue, get the HTML page, examine its
        links for download candidates and candidates for further scraping.

        This is a handy method to run in a thread.
        """
        while True:
            url = self._to_fetch.get()
            try:
                if url:
                    page = self.get_page(url)
                    if page is None:
                        continue
                    for link, rel in page.links:
                        if link not in self._seen:
                            try:
                                self._seen.add(link)
                                if not self._process_download(link) and self._should_queue(link, url, rel):
                                    logger.debug('Queueing %s from %s', link, url)
                                    self._to_fetch.put(link)
                            except MetadataInvalidError:
                                pass
            except Exception as e:
                self.errors.put(text_type(e))
            finally:
                self._to_fetch.task_done()
            if not url:
                break

    def get_page(self, url):
        """
        Get the HTML for an URL, possibly from an in-memory cache.

        XXX TODO Note: this cache is never actually cleared. It's assumed that
        the data won't get stale over the lifetime of a locator instance (not
        necessarily true for the default_locator).
        """
        scheme, netloc, path, _, _, _ = urlparse(url)
        if scheme == 'file' and os.path.isdir(url2pathname(path)):
            url = urljoin(ensure_slash(url), 'index.html')
        if url in self._page_cache:
            result = self._page_cache[url]
            logger.debug('Returning %s from cache: %s', url, result)
        else:
            host = netloc.split(':', 1)[0]
            result = None
            if host in self._bad_hosts:
                logger.debug('Skipping %s due to bad host %s', url, host)
            else:
                req = Request(url, headers={'Accept-encoding': 'identity'})
                try:
                    logger.debug('Fetching %s', url)
                    resp = self.opener.open(req, timeout=self.timeout)
                    logger.debug('Fetched %s', url)
                    headers = resp.info()
                    content_type = headers.get('Content-Type', '')
                    if HTML_CONTENT_TYPE.match(content_type):
                        final_url = resp.geturl()
                        data = resp.read()
                        encoding = headers.get('Content-Encoding')
                        if encoding:
                            decoder = self.decoders[encoding]
                            data = decoder(data)
                        encoding = 'utf-8'
                        m = CHARSET.search(content_type)
                        if m:
                            encoding = m.group(1)
                        try:
                            data = data.decode(encoding)
                        except UnicodeError:
                            data = data.decode('latin-1')
                        result = Page(data, final_url)
                        self._page_cache[final_url] = result
                except HTTPError as e:
                    if e.code != 404:
                        logger.exception('Fetch failed: %s: %s', url, e)
                except URLError as e:
                    logger.exception('Fetch failed: %s: %s', url, e)
                    with self._lock:
                        self._bad_hosts.add(host)
                except Exception as e:
                    logger.exception('Fetch failed: %s: %s', url, e)
                finally:
                    self._page_cache[url] = result
        return result
    _distname_re = re.compile('<a href=[^>]*>([^<]+)<')

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        result = set()
        page = self.get_page(self.base_url)
        if not page:
            raise DistlibException('Unable to get %s' % self.base_url)
        for match in self._distname_re.finditer(page.data):
            result.add(match.group(1))
        return result