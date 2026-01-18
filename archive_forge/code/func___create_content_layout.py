from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
@staticmethod
def __create_content_layout(layout_providers: list[t.Type[LayoutProvider]], source_providers: list[t.Type[SourceProvider]], root: str, walk: bool) -> t.Tuple[ContentLayout, SourceProvider]:
    """Create a content layout using the given providers and root path."""
    try:
        layout_provider = find_path_provider(LayoutProvider, layout_providers, root, walk)
    except ProviderNotFoundForPath:
        layout_provider = UnsupportedLayout(root)
    try:
        if isinstance(layout_provider, UnsupportedLayout):
            source_provider: SourceProvider = UnsupportedSource(layout_provider.root)
        else:
            source_provider = find_path_provider(SourceProvider, source_providers, layout_provider.root, walk)
    except ProviderNotFoundForPath:
        source_provider = UnversionedSource(layout_provider.root)
    layout = layout_provider.create(layout_provider.root, source_provider.get_paths(layout_provider.root))
    return (layout, source_provider)