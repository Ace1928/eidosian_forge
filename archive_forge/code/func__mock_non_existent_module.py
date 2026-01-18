import os
import sys
import mock
import pytest  # type: ignore
def _mock_non_existent_module(path):
    parts = path.split('.')
    partial = []
    for part in parts:
        partial.append(part)
        current_module = '.'.join(partial)
        if current_module not in sys.modules:
            monkeypatch.setitem(sys.modules, current_module, mock.MagicMock())