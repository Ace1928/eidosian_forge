from pyparsing import *
import random
import string
class DropCommand(Command):

    def __init__(self, quals):
        super(DropCommand, self).__init__('DROP', 'dropping')
        self.subject = quals.item

    @staticmethod
    def helpDescription():
        return 'DROP or LEAVE - drop an object (but fragile items may break)'

    def _doCommand(self, player):
        rm = player.room
        subj = Item.items[self.subject]
        if subj in player.inv:
            rm.addItem(subj)
            player.drop(subj)
        else:
            print("You don't have %s." % aOrAn(subj))