from __future__ import annotations
from typing import Any
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
def TransformEntryRootPath(unused_ref: str, args: Any, request: messages.DataplexProjectsLocationsLookupEntryRequest | messages.DataplexProjectsLocationsEntryGroupsEntriesGetRequest):
    """Transforms the root path from the "." in CLI to empty string expected in API."""
    if args.paths is not None and isinstance(args.paths, list):
        request.paths = list(set(map(lambda p: p if p != '.' else '', args.paths)))
    return request