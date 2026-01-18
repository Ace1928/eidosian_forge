from pyparsing import *
import random
import string
class UseCommand(Command):

    def __init__(self, quals):
        super(UseCommand, self).__init__('USE', 'using')
        self.subject = Item.items[quals.usedObj]
        if quals.targetObj:
            self.target = Item.items[quals.targetObj]
        else:
            self.target = None

    @staticmethod
    def helpDescription():
        return 'USE or U - use an object, optionally IN or ON another object'

    def _doCommand(self, player):
        rm = player.room
        availItems = rm.inv + player.inv
        if self.subject in availItems:
            if self.subject.isUsable(player, self.target):
                self.subject.useItem(player, self.target)
            else:
                print("You can't use that here.")
        else:
            print('There is no %s here to use.' % self.subject)