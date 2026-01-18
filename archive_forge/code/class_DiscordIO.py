import logging
from os import getenv
from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker
class DiscordIO(MonoWorker):
    """Non-blocking file-like IO using a Discord Bot."""

    def __init__(self, token, channel_id):
        """Creates a new message in the given `channel_id`."""
        super(DiscordIO, self).__init__()
        config = ClientConfig()
        config.token = token
        client = Client(config)
        self.text = self.__class__.__name__
        try:
            self.message = client.api.channels_messages_create(channel_id, self.text)
        except Exception as e:
            tqdm_auto.write(str(e))
            self.message = None

    def write(self, s):
        """Replaces internal `message`'s text with `s`."""
        if not s:
            s = '...'
        s = s.replace('\r', '').strip()
        if s == self.text:
            return
        message = self.message
        if message is None:
            return
        self.text = s
        try:
            future = self.submit(message.edit, '`' + s + '`')
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            return future