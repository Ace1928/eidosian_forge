#!/usr/bin/env python3
"""Process and system metrics from Linux /proc with graceful fallback."""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

try:  # optional psutil enrichment
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

__all__ = ["process_stats", "system_stats", "CpuPercent"]


def _read_proc_stat() -> tuple[float, float]:
    """Return process user and system CPU times in seconds."""
    clk = os.sysconf("SC_CLK_TCK")
    with open("/proc/self/stat", "r", encoding="utf-8") as f:
        parts = f.read().split()
    utime = float(parts[13]) / clk
    stime = float(parts[14]) / clk
    return utime, stime


def process_stats() -> Dict[str, Optional[float]]:
    """Return RSS and CPU times for current process."""
    out: Dict[str, Optional[float]] = {
        "rss_bytes": None,
        "cpu_user_s": None,
        "cpu_sys_s": None,
        "num_threads": None,
    }
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().split()
        rss_pages = int(parts[1])
        out["rss_bytes"] = rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        pass
    try:
        utime, stime = _read_proc_stat()
        out["cpu_user_s"] = utime
        out["cpu_sys_s"] = stime
    except OSError:
        pass
    if psutil is not None:
        try:
            proc = psutil.Process()
            out["num_threads"] = float(proc.num_threads())
        except Exception:
            pass
    return out


def system_stats() -> Dict[str, Optional[float]]:
    """Return basic system load and memory stats."""
    out: Dict[str, Optional[float]] = {
        "load1": None,
        "load5": None,
        "load15": None,
        "mem_total_kb": None,
        "mem_free_kb": None,
        "mem_available_kb": None,
        "cpu_pct_total": None,
        "swap_free_kb": None,
    }
    try:
        load1, load5, load15 = os.getloadavg()
        out.update({"load1": load1, "load5": load5, "load15": load15})
    except OSError:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            meminfo = {}
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meminfo[k.strip()] = v.strip().split()[0]
        for key, target in {
            "MemTotal": "mem_total_kb",
            "MemFree": "mem_free_kb",
            "MemAvailable": "mem_available_kb",
        }.items():
            if key in meminfo:
                out[target] = float(meminfo[key])
    except OSError:
        pass
    if psutil is not None:
        try:
            out["cpu_pct_total"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            swap = psutil.swap_memory()
            out["swap_free_kb"] = float(swap.free) / 1024
        except Exception:
            pass
    return out


class CpuPercent:
    """Compute CPU percentage for current process."""

    def __init__(self) -> None:
        self._last_wall: Optional[float] = None
        self._last_cpu: Optional[float] = None

    def sample(self) -> Optional[float]:
        try:
            utime, stime = _read_proc_stat()
        except OSError:
            return None
        now = time.monotonic()
        total = utime + stime
        if self._last_wall is None or self._last_cpu is None:
            self._last_wall = now
            self._last_cpu = total
            return None
        wall_delta = now - self._last_wall
        cpu_delta = total - self._last_cpu
        self._last_wall = now
        self._last_cpu = total
        if wall_delta <= 0:
            return 0.0
        pct = (cpu_delta / wall_delta) * 100.0
        if pct < 0:
            pct = 0.0
        if pct > 100.0:
            pct = 100.0
        return pct
