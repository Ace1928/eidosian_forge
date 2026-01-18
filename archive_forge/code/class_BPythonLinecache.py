import linecache
from typing import Any, List, Tuple, Optional
class BPythonLinecache(dict):
    """Replaces the cache dict in the standard-library linecache module,
    to also remember (in an unerasable way) bpython console input."""

    def __init__(self, bpython_history: Optional[List[Tuple[int, None, List[str], str]]]=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bpython_history = bpython_history or []

    def is_bpython_filename(self, fname: Any) -> bool:
        return isinstance(fname, str) and fname.startswith('<bpython-input-')

    def get_bpython_history(self, key: str) -> Tuple[int, None, List[str], str]:
        """Given a filename provided by remember_bpython_input,
        returns the associated source string."""
        try:
            idx = int(key.split('-')[2][:-1])
            return self.bpython_history[idx]
        except (IndexError, ValueError):
            raise KeyError

    def remember_bpython_input(self, source: str) -> str:
        """Remembers a string of source code, and returns
        a fake filename to use to retrieve it later."""
        filename = f'<bpython-input-{len(self.bpython_history)}>'
        self.bpython_history.append((len(source), None, source.splitlines(True), filename))
        return filename

    def __getitem__(self, key: Any) -> Any:
        if self.is_bpython_filename(key):
            return self.get_bpython_history(key)
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if self.is_bpython_filename(key):
            try:
                self.get_bpython_history(key)
                return True
            except KeyError:
                return False
        return super().__contains__(key)

    def __delitem__(self, key: Any) -> None:
        if not self.is_bpython_filename(key):
            super().__delitem__(key)