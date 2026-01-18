from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
class SilentListener(AbstractListener):

    def _set_volume(self, volume):
        pass

    def _set_position(self, position):
        pass

    def _set_forward_orientation(self, orientation):
        pass

    def _set_up_orientation(self, orientation):
        pass

    def _set_orientation(self):
        pass