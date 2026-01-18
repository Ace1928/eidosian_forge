import collections
import email.message
import functools
import itertools
import json
import logging
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from optparse import Values
from typing import (
from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.vcs import vcs
from .sources import CandidatesFromPage, LinkSource, build_source
class LinkCollector:
    """
    Responsible for collecting Link objects from all configured locations,
    making network requests as needed.

    The class's main method is its collect_sources() method.
    """

    def __init__(self, session: PipSession, search_scope: SearchScope) -> None:
        self.search_scope = search_scope
        self.session = session

    @classmethod
    def create(cls, session: PipSession, options: Values, suppress_no_index: bool=False) -> 'LinkCollector':
        """
        :param session: The Session to use to make requests.
        :param suppress_no_index: Whether to ignore the --no-index option
            when constructing the SearchScope object.
        """
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index and (not suppress_no_index):
            logger.debug('Ignoring indexes: %s', ','.join((redact_auth_from_url(url) for url in index_urls)))
            index_urls = []
        find_links = options.find_links or []
        search_scope = SearchScope.create(find_links=find_links, index_urls=index_urls, no_index=options.no_index)
        link_collector = LinkCollector(session=session, search_scope=search_scope)
        return link_collector

    @property
    def find_links(self) -> List[str]:
        return self.search_scope.find_links

    def fetch_response(self, location: Link) -> Optional[IndexContent]:
        """
        Fetch an HTML page containing package links.
        """
        return _get_index_content(location, session=self.session)

    def collect_sources(self, project_name: str, candidates_from_page: CandidatesFromPage) -> CollectedSources:
        index_url_sources = collections.OrderedDict((build_source(loc, candidates_from_page=candidates_from_page, page_validator=self.session.is_secure_origin, expand_dir=False, cache_link_parsing=False, project_name=project_name) for loc in self.search_scope.get_index_urls_locations(project_name))).values()
        find_links_sources = collections.OrderedDict((build_source(loc, candidates_from_page=candidates_from_page, page_validator=self.session.is_secure_origin, expand_dir=True, cache_link_parsing=True, project_name=project_name) for loc in self.find_links)).values()
        if logger.isEnabledFor(logging.DEBUG):
            lines = [f'* {s.link}' for s in itertools.chain(find_links_sources, index_url_sources) if s is not None and s.link is not None]
            lines = [f'{len(lines)} location(s) to search for versions of {project_name}:'] + lines
            logger.debug('\n'.join(lines))
        return CollectedSources(find_links=list(find_links_sources), index_urls=list(index_url_sources))