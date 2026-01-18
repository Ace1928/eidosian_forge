import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
class LessTest(unittest.TestCase):

    def setUp(self) -> None:
        self.cmd: command.PagerCommand = command.Less()

    def _env(self, config: command.PagerConfig) -> typing.Dict:
        return self.cmd.environment_variables(config) or {}

    def test_cmd(self) -> None:
        self.assertEqual(['less'], self.cmd.command())

    def test_less_env_defaults(self) -> None:
        config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=False)
        less_env = self._env(config)['LESS']
        self.assertEqual('RFX', less_env)

    def test_less_env_nocolor(self) -> None:
        config = command.PagerConfig(color=False, line_buffering_requested=False, reset_terminal=False)
        less_env = self._env(config)['LESS']
        self.assertEqual('FX', less_env)

    def test_less_env_reset(self) -> None:
        config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=True)
        less_env = self._env(config)['LESS']
        self.assertEqual('R', less_env)

    def test_less_env_nocolor_reset(self) -> None:
        config = command.PagerConfig(color=False, line_buffering_requested=False, reset_terminal=True)
        self.assertNotIn('LESS', self._env(config))

    def test_less_env_linebuffered(self) -> None:
        config = command.PagerConfig(color=True, line_buffering_requested=True, reset_terminal=False)
        less_env = self._env(config)['LESS']
        self.assertEqual('RX', less_env)

    def test_less_env_linebuffered_reset(self) -> None:
        config = command.PagerConfig(color=True, line_buffering_requested=True, reset_terminal=True)
        less_env = self._env(config)['LESS']
        self.assertEqual('R', less_env)

    def test_less_env_linebuffered_nocolor(self) -> None:
        config = command.PagerConfig(color=False, line_buffering_requested=True, reset_terminal=False)
        less_env = self._env(config)['LESS']
        self.assertEqual('X', less_env)

    def test_less_env_linebuffered_nocolor_reset(self) -> None:
        config = command.PagerConfig(color=False, line_buffering_requested=True, reset_terminal=True)
        self.assertNotIn('LESS', self._env(config))

    def test_less_env_override(self) -> None:
        config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=False)
        with fixtures.EnvironmentVariable('LESS', 'abc'):
            env = self._env(config)
        self.assertNotIn('LESS', env)