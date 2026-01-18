import typing
from typing import Optional
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher._sequencer import base
from google.pubsub_v1 import types as gapic_types
class UnorderedSequencer(base.Sequencer):
    """Sequences messages into batches for one topic without any ordering.

    Public methods are NOT thread-safe.
    """

    def __init__(self, client: 'PublisherClient', topic: str):
        self._client = client
        self._topic = topic
        self._current_batch: Optional['_batch.thread.Batch'] = None
        self._stopped = False

    def is_finished(self) -> bool:
        """Whether the sequencer is finished and should be cleaned up.

        Returns:
            Whether the sequencer is finished and should be cleaned up.
        """
        return False

    def stop(self) -> None:
        """Stop the sequencer.

        Subsequent publishes will fail.

        Raises:
            RuntimeError:
                If called after stop() has already been called.
        """
        if self._stopped:
            raise RuntimeError('Unordered sequencer already stopped.')
        self.commit()
        self._stopped = True

    def commit(self) -> None:
        """Commit the batch.

        Raises:
            RuntimeError:
                If called after stop() has already been called.
        """
        if self._stopped:
            raise RuntimeError('Unordered sequencer already stopped.')
        if self._current_batch:
            self._current_batch.commit()
            self._current_batch = None

    def unpause(self) -> typing.NoReturn:
        """Not relevant for this class."""
        raise NotImplementedError

    def _create_batch(self, commit_retry: 'OptionalRetry'=gapic_v1.method.DEFAULT, commit_timeout: 'types.OptionalTimeout'=gapic_v1.method.DEFAULT) -> '_batch.thread.Batch':
        """Create a new batch using the client's batch class and other stored
            settings.

        Args:
            commit_retry:
                The retry settings to apply when publishing the batch.
            commit_timeout:
                The timeout to apply when publishing the batch.
        """
        return self._client._batch_class(client=self._client, topic=self._topic, settings=self._client.batch_settings, batch_done_callback=None, commit_when_full=True, commit_retry=commit_retry, commit_timeout=commit_timeout)

    def publish(self, message: gapic_types.PubsubMessage, retry: 'OptionalRetry'=gapic_v1.method.DEFAULT, timeout: 'types.OptionalTimeout'=gapic_v1.method.DEFAULT) -> 'futures.Future':
        """Batch message into existing or new batch.

        Args:
            message:
                The Pub/Sub message.
            retry:
                The retry settings to apply when publishing the message.
            timeout:
                The timeout to apply when publishing the message.

        Returns:
            An object conforming to the :class:`~concurrent.futures.Future` interface.
            The future tracks the publishing status of the message.

        Raises:
            RuntimeError:
                If called after stop() has already been called.

            pubsub_v1.publisher.exceptions.MessageTooLargeError: If publishing
                the ``message`` would exceed the max size limit on the backend.
        """
        if self._stopped:
            raise RuntimeError('Unordered sequencer already stopped.')
        if not self._current_batch:
            newbatch = self._create_batch(commit_retry=retry, commit_timeout=timeout)
            self._current_batch = newbatch
        batch = self._current_batch
        future = None
        while future is None:
            future = batch.publish(message)
            if future is None:
                batch = self._create_batch(commit_retry=retry, commit_timeout=timeout)
                self._current_batch = batch
        return future

    def _set_batch(self, batch: '_batch.thread.Batch') -> None:
        self._current_batch = batch