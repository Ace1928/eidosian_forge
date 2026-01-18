from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
class PlayerGroup:
    """Group of players that can be played and paused simultaneously.

    Create a player group for the given list of players.

    All players in the group must currently not belong to any other group.

    Args:
        players (Iterable[Player]): Iterable of :class:`.Player` s in this
            group.
    """

    def __init__(self, players: Iterable[Player]) -> None:
        """Initialize the PlayerGroup with the players."""
        self.players = list(players)

    def play(self) -> None:
        """Begin playing all players in the group simultaneously."""
        audio_players = [p._audio_player for p in self.players if p._audio_player]
        if audio_players:
            audio_players[0]._play_group(audio_players)
        for player in self.players:
            player.play()

    def pause(self) -> None:
        """Pause all players in the group simultaneously."""
        audio_players = [p._audio_player for p in self.players if p._audio_player]
        if audio_players:
            audio_players[0]._stop_group(audio_players)
        for player in self.players:
            player.pause()