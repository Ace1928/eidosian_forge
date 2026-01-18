import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class FakeCommandManager(commandmanager.CommandManager):
    commands = {}

    def load_commands(self, namespace):
        if namespace == 'test':
            self.commands['one'] = FAKE_CMD_ONE
            self.commands['two'] = FAKE_CMD_TWO
            self.group_list.append(namespace)
        elif namespace == 'greek':
            self.commands['alpha'] = FAKE_CMD_ALPHA
            self.commands['beta'] = FAKE_CMD_BETA
            self.group_list.append(namespace)