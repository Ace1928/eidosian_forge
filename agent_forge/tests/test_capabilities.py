from agent_forge.core.capabilities import Capabilities
from code_forge.library.db import CodeLibraryDB, CodeUnit


def test_capabilities(tmp_path):
    # Setup dummy file
    f = tmp_path / "test.py"
    f.write_text("def hello(): pass")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    db.add_unit(
        CodeUnit(
            unit_type="function",
            name="hello",
            qualified_name="hello",
            file_path=str(f),
            semantic_text="hello greeting function",
        )
    )
    caps = Capabilities(repo_root=tmp_path, library_db=db)

    # Analyze
    res = caps.analyze_code(str(f))
    assert len(res["functions"]) == 1
    assert res["functions"][0]["name"] == "hello"

    hits = caps.search_code("greeting function", limit=5)
    assert hits
    assert hits[0]["name"] == "hello"

    tools = caps.get_tool_map()
    assert "analyze_code" in tools
    assert "search_code" in tools
