from __future__ import annotations
import logging
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from pymatgen.core import DummySpecies, Structure
from pymatgen.util.due import Doi, due
from pymatgen.util.provenance import StructureNL
def _validate_provider(self, provider_url) -> Provider | None:
    """Checks that a given URL is indeed an OPTIMADE provider,
        returning None if it is not a provider, or the provider
        prefix if it is.

        TODO: careful reading of OPTIMADE specification required
        TODO: add better exception handling, intentionally permissive currently
        """

    def is_url(url) -> bool:
        """Basic URL validation thanks to https://stackoverflow.com/a/52455972."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    if not is_url(provider_url):
        _logger.warning(f'An invalid url was supplied: {provider_url}')
        return None
    try:
        url = urljoin(provider_url, 'v1/info')
        provider_info_json = self._get_json(url)
    except Exception as exc:
        _logger.warning(f'Failed to parse {url} when validating: {exc}')
        return None
    try:
        return Provider(name=provider_info_json['meta'].get('provider', {}).get('name', 'Unknown'), base_url=provider_url, description=provider_info_json['meta'].get('provider', {}).get('description', 'Unknown'), homepage=provider_info_json['meta'].get('provider', {}).get('homepage'), prefix=provider_info_json['meta'].get('provider', {}).get('prefix', 'Unknown'))
    except Exception as exc:
        _logger.warning(f'Failed to extract required information from {url}: {exc}')
        return None