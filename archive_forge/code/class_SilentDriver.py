from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
class SilentDriver(AbstractAudioDriver):

    def __init__(self) -> None:
        super().__init__()
        self.worker = PlayerWorkerThread()
        self.worker.start()

    def create_audio_player(self, source, player):
        return SilentAudioPlayer(self, source, player)

    def get_listener(self):
        return SilentListener()

    def delete(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None