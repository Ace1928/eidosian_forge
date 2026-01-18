import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
def _on_player_eos():
    Source._players.remove(player)
    player.on_player_eos = None
    player.delete()