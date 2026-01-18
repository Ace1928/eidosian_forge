from pyparsing import *
import random
import string
class CloseCommand(Command):

    def __init__(self, quals):
        super(CloseCommand, self).__init__('CLOSE', 'closing')
        self.subject = Item.items[quals.item]

    @staticmethod
    def helpDescription():
        return 'CLOSE or CL - close an object'

    def _doCommand(self, player):
        rm = player.room
        availItems = rm.inv + player.inv
        if self.subject in availItems:
            if self.subject.isOpenable:
                if self.subject.isOpened:
                    self.subject.closeItem(player)
                else:
                    print("You can't close that, it's not open.")
            else:
                print("You can't close that.")
        else:
            print('There is no %s here to close.' % self.subject)