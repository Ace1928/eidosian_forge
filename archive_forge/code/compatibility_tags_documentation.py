import re
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import (
Return a list of supported tags for each version specified in
    `versions`.

    :param version: a string version, of the form "33" or "32",
        or None. The version will be assumed to support our ABI.
    :param platform: specify a list of platforms you want valid
        tags for, or None. If None, use the local system platform.
    :param impl: specify the exact implementation you want valid
        tags for, or None. If None, use the local interpreter impl.
    :param abis: specify a list of abis you want valid
        tags for, or None. If None, use the local interpreter abi.
    