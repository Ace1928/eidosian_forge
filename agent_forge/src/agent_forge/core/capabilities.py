from eidosian_core import eidosian

"""
Agent Capabilities (Tools).
Integration with other Forges.
"""
from pathlib import Path
from typing import Any, Callable, Dict

from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.librarian.core import CodeLibrarian


class Capabilities:
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.librarian = CodeLibrarian(Path("./data/code_lib.json"))

    @eidosian()
    def analyze_code(self, file_path: str) -> Dict[str, Any]:
        p = Path(file_path)
        if not p.exists():
            return {"error": "File not found"}
        return self.code_analyzer.analyze_file(p)

    @eidosian()
    def search_code(self, query: str) -> list:
        return self.librarian.search(query)

    @eidosian()
    def get_tool_map(self) -> Dict[str, Callable]:
        return {"analyze_code": self.analyze_code, "search_code": self.search_code}
