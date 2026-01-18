import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
def _compare_equal(self, prospective: Version, spec: str) -> bool:
    if spec.endswith('.*'):
        normalized_prospective = canonicalize_version(prospective.public, strip_trailing_zero=False)
        normalized_spec = canonicalize_version(spec[:-2], strip_trailing_zero=False)
        split_spec = _version_split(normalized_spec)
        split_prospective = _version_split(normalized_prospective)
        padded_prospective, _ = _pad_version(split_prospective, split_spec)
        shortened_prospective = padded_prospective[:len(split_spec)]
        return shortened_prospective == split_spec
    else:
        spec_version = Version(spec)
        if not spec_version.local:
            prospective = Version(prospective.public)
        return prospective == spec_version