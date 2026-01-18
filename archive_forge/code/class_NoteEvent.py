from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class NoteEvent(ChannelEvent):
    """
    NoteEvent is a special subclass of Event that is not meant to be used as a
    concrete class. It defines the generalities of NoteOn and NoteOff events.

    """
    length = 2

    def __str__(self):
        return '%s: tick: %s channel: %s pitch: %s velocity: %s' % (self.__class__.__name__, self.tick, self.channel, self.pitch, self.velocity)

    @property
    def pitch(self):
        """
        Pitch of the note event.

        """
        return self.data[0]

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch of the note event.

        Parameters
        ----------
        pitch : int
            Pitch of the note.

        """
        self.data[0] = pitch

    @property
    def velocity(self):
        """
        Velocity of the note event.

        """
        return self.data[1]

    @velocity.setter
    def velocity(self, velocity):
        """
        Set the velocity of the note event.

        Parameters
        ----------
        velocity : int
            Velocity of the note.

        """
        self.data[1] = velocity