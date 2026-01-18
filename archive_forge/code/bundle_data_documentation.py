import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
Build up the inventory entry for the BundleTree.

        This need to be called before ever accessing self.inventory
        