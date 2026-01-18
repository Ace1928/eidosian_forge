import unittest
import fixtures  # type: ignore
from typing import Any, Optional, Dict, List
import autopage
from autopage import command
def _get_ap_config(self, **args: Any) -> command.PagerConfig:
    ap = autopage.AutoPager(pager_command=self.test_command, **args)
    ap._pager_env()
    config = self.test_command.config
    assert config is not None
    return config