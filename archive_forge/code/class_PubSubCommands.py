import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class PubSubCommands(CommandsProtocol):
    """
    Redis PubSub commands.
    see https://redis.io/topics/pubsub
    """

    def publish(self, channel: ChannelT, message: EncodableT, **kwargs) -> ResponseT:
        """
        Publish ``message`` on ``channel``.
        Returns the number of subscribers the message was delivered to.

        For more information see https://redis.io/commands/publish
        """
        return self.execute_command('PUBLISH', channel, message, **kwargs)

    def spublish(self, shard_channel: ChannelT, message: EncodableT) -> ResponseT:
        """
        Posts a message to the given shard channel.
        Returns the number of clients that received the message

        For more information see https://redis.io/commands/spublish
        """
        return self.execute_command('SPUBLISH', shard_channel, message)

    def pubsub_channels(self, pattern: PatternT='*', **kwargs) -> ResponseT:
        """
        Return a list of channels that have at least one subscriber

        For more information see https://redis.io/commands/pubsub-channels
        """
        return self.execute_command('PUBSUB CHANNELS', pattern, **kwargs)

    def pubsub_shardchannels(self, pattern: PatternT='*', **kwargs) -> ResponseT:
        """
        Return a list of shard_channels that have at least one subscriber

        For more information see https://redis.io/commands/pubsub-shardchannels
        """
        return self.execute_command('PUBSUB SHARDCHANNELS', pattern, **kwargs)

    def pubsub_numpat(self, **kwargs) -> ResponseT:
        """
        Returns the number of subscriptions to patterns

        For more information see https://redis.io/commands/pubsub-numpat
        """
        return self.execute_command('PUBSUB NUMPAT', **kwargs)

    def pubsub_numsub(self, *args: ChannelT, **kwargs) -> ResponseT:
        """
        Return a list of (channel, number of subscribers) tuples
        for each channel given in ``*args``

        For more information see https://redis.io/commands/pubsub-numsub
        """
        return self.execute_command('PUBSUB NUMSUB', *args, **kwargs)

    def pubsub_shardnumsub(self, *args: ChannelT, **kwargs) -> ResponseT:
        """
        Return a list of (shard_channel, number of subscribers) tuples
        for each channel given in ``*args``

        For more information see https://redis.io/commands/pubsub-shardnumsub
        """
        return self.execute_command('PUBSUB SHARDNUMSUB', *args, **kwargs)