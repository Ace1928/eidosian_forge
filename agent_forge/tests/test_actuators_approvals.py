from actuators.approvals import allowed_cmd

def test_allow_and_deny(tmp_path):
    ok, _ = allowed_cmd(["pytest","-q"], str(tmp_path))
    assert ok
    ok, reason = allowed_cmd(["curl","http://x"], str(tmp_path))
    assert not ok and "denied" in reason

def test_cwd_scope(tmp_path):
    ok, reason = allowed_cmd(["pytest","-q"], "/")
    assert not ok and "cwd escapes" in reason
