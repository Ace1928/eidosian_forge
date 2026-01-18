from pyparsing import *
import random
import string
class OpenableItem(Item):

    def __init__(self, desc, contents=None):
        super(OpenableItem, self).__init__(desc)
        self.isOpenable = True
        self.isOpened = False
        if contents is not None:
            if isinstance(contents, Item):
                self.contents = [contents]
            else:
                self.contents = contents
        else:
            self.contents = []

    def openItem(self, player):
        if not self.isOpened:
            self.isOpened = not self.isOpened
            if self.contents is not None:
                for item in self.contents:
                    player.room.addItem(item)
                self.contents = []
            self.desc = 'open ' + self.desc

    def closeItem(self, player):
        if self.isOpened:
            self.isOpened = not self.isOpened
            if self.desc.startswith('open '):
                self.desc = self.desc[5:]