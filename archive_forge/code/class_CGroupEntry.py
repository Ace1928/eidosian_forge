from __future__ import annotations
import codecs
import dataclasses
import pathlib
import re
@dataclasses.dataclass(frozen=True)
class CGroupEntry:
    """A single cgroup entry parsed from '/proc/{pid}/cgroup' in the proc filesystem."""
    id: int
    subsystem: str
    path: pathlib.PurePosixPath

    @property
    def root_path(self) -> pathlib.PurePosixPath:
        """The root path for this cgroup subsystem."""
        return pathlib.PurePosixPath(CGroupPath.ROOT, self.subsystem)

    @property
    def full_path(self) -> pathlib.PurePosixPath:
        """The full path for this cgroup subsystem."""
        return pathlib.PurePosixPath(self.root_path, str(self.path).lstrip('/'))

    @classmethod
    def parse(cls, value: str) -> CGroupEntry:
        """Parse the given cgroup line from the proc filesystem and return a cgroup entry."""
        cid, subsystem, path = value.split(':', maxsplit=2)
        return cls(id=int(cid), subsystem=subsystem.removeprefix('name='), path=pathlib.PurePosixPath(path))

    @classmethod
    def loads(cls, value: str) -> tuple[CGroupEntry, ...]:
        """Parse the given output from the proc filesystem and return a tuple of cgroup entries."""
        return tuple((cls.parse(line) for line in value.splitlines()))