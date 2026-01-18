from pyparsing import *
import random
import string
class DoorsCommand(Command):

    def __init__(self, quals):
        super(DoorsCommand, self).__init__('DOORS', 'looking for doors')

    @staticmethod
    def helpDescription():
        return 'DOORS - display what doors are visible from this room'

    def _doCommand(self, player):
        rm = player.room
        numDoors = sum([1 for r in rm.doors if r is not None])
        if numDoors == 0:
            reply = 'There are no doors in any direction.'
        else:
            if numDoors == 1:
                reply = 'There is a door to the '
            else:
                reply = 'There are doors to the '
            doorNames = [{0: 'north', 1: 'south', 2: 'east', 3: 'west'}[i] for i, d in enumerate(rm.doors) if d is not None]
            reply += enumerateDoors(doorNames)
            reply += '.'
            print(reply)