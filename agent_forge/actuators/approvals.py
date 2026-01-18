from __future__ import annotations
from pathlib import Path
from typing import Sequence

_DEFAULT_ALLOW = ["uv","pytest","ruff","black","git","bash","sh","make","python","pip"]
_DEFAULT_DENY_TOKENS = ["curl","wget","ssh","nc","iptables","docker"]

try:
    import yaml
except Exception:
    yaml = None

_CFG_PATH = Path("cfg/approvals.yaml")
if yaml and _CFG_PATH.exists():
    try:
        _CFG = yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        _CFG = {}
else:
    _CFG = {}


def allowed_cmd(cmd: Sequence[str], cwd: str, *, template: str | None = None,
                allow_prefixes=None, deny_tokens=None) -> tuple[bool, str]:
    """Return (ok, reason). Enforces prefix allowlist and deny token scan.
       CWD must be inside repo (no path escapes)."""
    if not cmd:
        return False, "empty command"
    defaults = _CFG.get("defaults", {})
    allow = list(allow_prefixes or defaults.get("allow_prefixes", _DEFAULT_ALLOW))
    deny = set(deny_tokens or defaults.get("deny_tokens", _DEFAULT_DENY_TOKENS))
    if template:
        tpl = (_CFG.get("templates", {}) or {}).get(template, {})
        if "allow_prefixes" in tpl:
            allow = list(tpl["allow_prefixes"])
        if "deny_tokens" in tpl:
            deny = set(tpl["deny_tokens"])
    flat = " ".join(cmd)
    if any(t in flat for t in deny):
        return False, "denied token present"
    exe = Path(cmd[0]).name
    if not any(exe.startswith(p) for p in allow):
        return False, f"disallowed exe: {exe}"
    # scope cwd: disallow running from filesystem root
    here = Path(cwd).resolve()
    if here == Path("/"):
        return False, "cwd escapes repo"
    return True, "ok"
