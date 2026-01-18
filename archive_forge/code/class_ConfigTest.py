import unittest
import fixtures  # type: ignore
from typing import Any, Optional, Dict, List
import autopage
from autopage import command
class ConfigTest(unittest.TestCase):

    def setUp(self) -> None:

        class TestCommand(command.PagerCommand):

            def __init__(self) -> None:
                self.config: Optional[_PagerConfig] = None

            def command(self) -> List[str]:
                return []

            def environment_variables(self, config: _PagerConfig) -> Optional[Dict[str, str]]:
                self.config = config
                return None
        self.test_command = TestCommand()

    def _get_ap_config(self, **args: Any) -> command.PagerConfig:
        ap = autopage.AutoPager(pager_command=self.test_command, **args)
        ap._pager_env()
        config = self.test_command.config
        assert config is not None
        return config

    def test_defaults(self) -> None:
        config = self._get_ap_config()
        self.assertTrue(config.color)
        self.assertFalse(config.line_buffering_requested)
        self.assertFalse(config.reset_terminal)

    def test_nocolor(self) -> None:
        config = self._get_ap_config(allow_color=False)
        self.assertFalse(config.color)
        self.assertFalse(config.line_buffering_requested)
        self.assertFalse(config.reset_terminal)

    def test_reset(self) -> None:
        config = self._get_ap_config(reset_on_exit=True)
        self.assertTrue(config.color)
        self.assertFalse(config.line_buffering_requested)
        self.assertTrue(config.reset_terminal)

    def test_linebuffered(self) -> None:
        config = self._get_ap_config(line_buffering=True)
        self.assertTrue(config.color)
        self.assertTrue(config.line_buffering_requested)
        self.assertFalse(config.reset_terminal)

    def test_not_linebuffered(self) -> None:
        config = self._get_ap_config(line_buffering=False)
        self.assertTrue(config.color)
        self.assertFalse(config.line_buffering_requested)
        self.assertFalse(config.reset_terminal)