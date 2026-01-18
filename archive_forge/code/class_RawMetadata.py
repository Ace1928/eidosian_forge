import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import Dict, List, Optional, Tuple, Union, cast
class RawMetadata(TypedDict, total=False):
    """A dictionary of raw core metadata.

    Each field in core metadata maps to a key of this dictionary (when data is
    provided). The key is lower-case and underscores are used instead of dashes
    compared to the equivalent core metadata field. Any core metadata field that
    can be specified multiple times or can hold multiple values in a single
    field have a key with a plural name.

    Core metadata fields that can be specified multiple times are stored as a
    list or dict depending on which is appropriate for the field. Any fields
    which hold multiple values in a single field are stored as a list.

    """
    metadata_version: str
    name: str
    version: str
    platforms: List[str]
    summary: str
    description: str
    keywords: List[str]
    home_page: str
    author: str
    author_email: str
    license: str
    supported_platforms: List[str]
    download_url: str
    classifiers: List[str]
    requires: List[str]
    provides: List[str]
    obsoletes: List[str]
    maintainer: str
    maintainer_email: str
    requires_dist: List[str]
    provides_dist: List[str]
    obsoletes_dist: List[str]
    requires_python: str
    requires_external: List[str]
    project_urls: Dict[str, str]
    description_content_type: str
    provides_extra: List[str]
    dynamic: List[str]