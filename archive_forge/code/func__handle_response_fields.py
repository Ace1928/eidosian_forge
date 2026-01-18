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
def _handle_response_fields(self, additional_response_fields: str | list[str] | set[str] | None=None) -> str:
    """Used internally to handle the mandatory and additional response fields.

        Args:
            additional_response_fields: A set of additional fields to request.

        Returns:
            A string of comma-separated OPTIMADE response fields.
        """
    if isinstance(additional_response_fields, str):
        additional_response_fields = {additional_response_fields}
    if not additional_response_fields:
        additional_response_fields = set()
    return ','.join({*additional_response_fields, *self.mandatory_response_fields})