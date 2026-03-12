from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.modules import knowledge_bridge, memory_bridge
from agent_forge.core import memory as core_memory
from agent_forge.core import self_model


def test_memory_path_defaults_prefer_tiered_memory(monkeypatch, tmp_path: Path) -> None:
    forge_root = tmp_path / "forge"
    tiered = forge_root / "data" / "tiered_memory"
    legacy = forge_root / "data" / "memory"
    tiered.mkdir(parents=True)
    legacy.mkdir(parents=True)

    monkeypatch.delenv("EIDOS_MEMORY_DIR", raising=False)
    monkeypatch.setattr(memory_bridge, "_forge_root", lambda: forge_root)
    monkeypatch.setattr(knowledge_bridge, "_forge_root", lambda: forge_root)
    monkeypatch.setattr(core_memory, "_forge_root", lambda: forge_root)
    monkeypatch.setattr(self_model, "FORGE_ROOT", forge_root)

    class _Dummy:
        memory_path = forge_root / "memory-root"

    assert memory_bridge._default_memory_dir() == tiered
    assert knowledge_bridge._default_memory_dir() == tiered
    assert self_model._default_memory_dir() == tiered
    assert core_memory.MemorySystem._tiered_memory_dir(_Dummy()) == tiered
