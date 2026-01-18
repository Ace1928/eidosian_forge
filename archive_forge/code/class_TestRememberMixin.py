from typing import List
from breezy import branch, urlutils
from breezy.tests import script
class TestRememberMixin:
    """--remember and --no-remember set locations or not."""
    command: List[str] = []
    working_dir: str
    first_use_args: List[str] = []
    next_uses_args: List[str] = []

    def do_command(self, *args):
        out, err = self.run_bzr(self.command + list(args), working_dir=self.working_dir)

    def test_first_use_no_option(self):
        self.do_command(*self.first_use_args)
        self.assertLocations(self.first_use_args)

    def test_first_use_remember(self):
        self.do_command('--remember', *self.first_use_args)
        self.assertLocations(self.first_use_args)

    def test_first_use_no_remember(self):
        self.do_command('--no-remember', *self.first_use_args)
        self.assertLocations([])

    def test_next_uses_no_option(self):
        self.setup_next_uses()
        self.do_command(*self.next_uses_args)
        self.assertLocations(self.first_use_args)

    def test_next_uses_remember(self):
        self.setup_next_uses()
        self.do_command('--remember', *self.next_uses_args)
        self.assertLocations(self.next_uses_args)

    def test_next_uses_no_remember(self):
        self.setup_next_uses()
        self.do_command('--no-remember', *self.next_uses_args)
        self.assertLocations(self.first_use_args)