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
def _parse_provider(self, provider: str, provider_url: str) -> dict[str, Provider]:
    """Used internally to update the list of providers or to
        check a given URL is valid.

        It does not raise exceptions but will instead _logger.warning and provide
        an empty dictionary in the case of invalid data.

        In future, when the specification  is sufficiently well adopted,
        we might be more strict here.

        Args:
            provider: the provider prefix
            provider_url: An OPTIMADE provider URL

        Returns:
            A dictionary of keys (in format of "provider.database") to
            Provider objects.
        """
    try:
        url = urljoin(provider_url, 'v1/links')
        provider_link_json = self._get_json(url)
    except Exception as exc:
        _logger.error(f'Failed to parse {url} when following links: {exc}')
        return {}

    def _parse_provider_link(provider, provider_link_json):
        """No validation attempted."""
        ps = {}
        try:
            data = [dct for dct in provider_link_json['data'] if dct['attributes']['link_type'] == 'child']
            for link in data:
                key = f'{provider}.{link['id']}' if provider != link['id'] else provider
                if link['attributes']['base_url']:
                    ps[key] = Provider(name=link['attributes']['name'], base_url=link['attributes']['base_url'], description=link['attributes']['description'], homepage=link['attributes'].get('homepage'), prefix=link['attributes'].get('prefix'))
        except Exception:
            pass
        return ps
    return _parse_provider_link(provider, provider_link_json)