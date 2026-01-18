from pyparsing import *
import random
import string
def _doCommand(self, player):
    print('Enter any of the following commands (not case sensitive):')
    for cmd in [InventoryCommand, DropCommand, TakeCommand, UseCommand, OpenCommand, CloseCommand, MoveCommand, LookCommand, DoorsCommand, QuitCommand, HelpCommand]:
        print('  - %s' % cmd.helpDescription())
    print()