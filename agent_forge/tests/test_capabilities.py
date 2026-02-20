

from agent_forge.core.capabilities import Capabilities


def test_capabilities(tmp_path):
    # Setup dummy file
    f = tmp_path / "test.py"
    f.write_text("def hello(): pass")

    caps = Capabilities()

    # Analyze
    res = caps.analyze_code(str(f))
    assert len(res["functions"]) == 1
    assert res["functions"][0]["name"] == "hello"

    # Search (Mocking librarian persistence path or just ensuring interface works)
    # The default path is ./data/code_lib.json which might not exist.
    # The test should ideally inject path.
    # But for now, just checking method existence/signature match.
    tools = caps.get_tool_map()
    assert "analyze_code" in tools
