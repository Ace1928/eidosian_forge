from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
 Break a block into list items. 