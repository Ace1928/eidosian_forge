from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


_STATE_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ModuleStateStore:
    """
    Persistent, namespaced state for consciousness modules.

    State is stored under:
      <state_dir>/consciousness/module_state.json
    """

    def __init__(
        self, state_dir: str | Path, *, autosave_interval_secs: float = 2.0
    ) -> None:
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "consciousness" / "module_state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.autosave_interval_secs = max(0.0, float(autosave_interval_secs))
        self._data: Dict[str, Any] = self._default_payload()
        self._dirty = False
        self._last_save_monotonic = 0.0
        self._load()

    def _default_payload(self) -> Dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "updated_at": _now_iso(),
            "meta": {
                "beat_count": 0,
            },
            "modules": {},
        }

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, Mapping):
            return

        modules = payload.get("modules")
        meta = payload.get("meta")
        if not isinstance(modules, Mapping):
            modules = {}
        if not isinstance(meta, Mapping):
            meta = {}

        self._data = {
            "version": int(payload.get("version") or _STATE_VERSION),
            "updated_at": str(payload.get("updated_at") or _now_iso()),
            "meta": dict(meta),
            "modules": {
                str(k): dict(v) for k, v in modules.items() if isinstance(v, Mapping)
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data, default=str))

    def namespace(
        self,
        module_name: str,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> MutableMapping[str, Any]:
        modules = self._data.setdefault("modules", {})
        if not isinstance(modules, MutableMapping):
            modules = {}
            self._data["modules"] = modules
        module_key = str(module_name)
        ns = modules.get(module_key)
        if not isinstance(ns, MutableMapping):
            ns = {}
            modules[module_key] = ns
            self._dirty = True

        if defaults:
            for key, value in defaults.items():
                if key not in ns:
                    ns[key] = value
                    self._dirty = True
        return ns

    def get_meta(self, key: str, default: Any = None) -> Any:
        meta = self._data.get("meta")
        if not isinstance(meta, Mapping):
            return default
        return meta.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        meta = self._data.setdefault("meta", {})
        if not isinstance(meta, MutableMapping):
            meta = {}
            self._data["meta"] = meta
        if meta.get(key) != value:
            meta[key] = value
            self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def flush(self, *, force: bool = False) -> bool:
        if not self._dirty and not force:
            return False

        now = time.monotonic()
        if not force and self.autosave_interval_secs > 0.0:
            if (now - self._last_save_monotonic) < self.autosave_interval_secs:
                return False

        self._data["version"] = _STATE_VERSION
        self._data["updated_at"] = _now_iso()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = False
        self._last_save_monotonic = now
        return True
