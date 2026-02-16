from __future__ import annotations

import uuid
from typing import Any, Iterable, Mapping


_LINK_KEYS = (
    "corr_id",
    "parent_id",
    "memory_ids",
    "winner_candidate_id",
    "candidate_id",
)


def new_corr_id(seed: str | None = None) -> str:
    """
    Generate a correlation id for event-thread linkage.

    When a seed is provided, use deterministic UUIDv5 so tests/replays can
    produce stable IDs from stable seeds.
    """
    if seed:
        return uuid.uuid5(uuid.NAMESPACE_OID, str(seed)).hex
    return uuid.uuid4().hex


def _merge_memory_ids(*values: Iterable[Any] | None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for items in values:
        if items is None:
            continue
        for item in items:
            text = str(item or "")
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def canonical_links(
    raw_links: Mapping[str, Any] | None,
    *,
    corr_id: str | None = None,
    parent_id: str | None = None,
    memory_ids: Iterable[Any] | None = None,
    candidate_id: str | None = None,
    winner_candidate_id: str | None = None,
) -> dict[str, Any]:
    links = dict(raw_links or {})
    merged_memory_ids = _merge_memory_ids(
        links.get("memory_ids") if isinstance(links.get("memory_ids"), list) else [],
        memory_ids,
    )

    out = {
        "corr_id": str(corr_id or links.get("corr_id") or ""),
        "parent_id": str(parent_id or links.get("parent_id") or ""),
        "memory_ids": merged_memory_ids,
        "winner_candidate_id": str(
            winner_candidate_id or links.get("winner_candidate_id") or ""
        ),
        "candidate_id": str(candidate_id or links.get("candidate_id") or ""),
    }

    # Preserve unknown link fields for forward compatibility.
    for key, value in links.items():
        if key in _LINK_KEYS:
            continue
        out[key] = value
    return out


def payload_link_candidates(payload: Mapping[str, Any]) -> tuple[str, str]:
    """
    Infer candidate linkage from payload content/kind when explicit fields are absent.
    """
    kind = str(payload.get("kind") or "")
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    candidate_id = str(content.get("candidate_id") or "")
    winner_candidate_id = str(content.get("winner_candidate_id") or "")

    if kind == "GW_WINNER":
        winner_candidate_id = winner_candidate_id or candidate_id
    if kind == "REPORT" and not winner_candidate_id:
        summary = (
            content.get("summary") if isinstance(content.get("summary"), Mapping) else {}
        )
        winner_candidate_id = str(summary.get("winner_candidate_id") or "")
    return candidate_id, winner_candidate_id
