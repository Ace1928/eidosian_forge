import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def _get_max_pdf_version_header(header1: str, header2: str) -> str:
    versions = ('%PDF-1.3', '%PDF-1.4', '%PDF-1.5', '%PDF-1.6', '%PDF-1.7', '%PDF-2.0')
    pdf_header_indices = []
    if header1 in versions:
        pdf_header_indices.append(versions.index(header1))
    if header2 in versions:
        pdf_header_indices.append(versions.index(header2))
    if len(pdf_header_indices) == 0:
        raise ValueError(f'neither {header1!r} nor {header2!r} are proper headers')
    return versions[max(pdf_header_indices)]