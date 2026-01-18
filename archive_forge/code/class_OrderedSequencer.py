import enum
import collections
import threading
import typing
from typing import Deque, Iterable, Sequence
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher import futures
from google.cloud.pubsub_v1.publisher import exceptions
from google.cloud.pubsub_v1.publisher._sequencer import base as sequencer_base
from google.cloud.pubsub_v1.publisher._batch import base as batch_base
from google.pubsub_v1 import types as gapic_types
class OrderedSequencer(sequencer_base.Sequencer):
    """Sequences messages into batches ordered by an ordering key for one topic.

    A sequencer always has at least one batch in it, unless paused or stopped.
    When no batches remain, the |publishes_done_callback| is called so the
    client can perform cleanup.

    Public methods are thread-safe.

    Args:
        client:
            The publisher client used to create this sequencer.
        topic:
            The topic. The format for this is ``projects/{project}/topics/{topic}``.
        ordering_key:
            The ordering key for this sequencer.
    """

    def __init__(self, client: 'PublisherClient', topic: str, ordering_key: str):
        self._client = client
        self._topic = topic
        self._ordering_key = ordering_key
        self._state_lock = threading.Lock()
        self._ordered_batches: Deque['_batch.thread.Batch'] = collections.deque()
        self._state = _OrderedSequencerStatus.ACCEPTING_MESSAGES

    def is_finished(self) -> bool:
        """Whether the sequencer is finished and should be cleaned up.

        Returns:
            Whether the sequencer is finished and should be cleaned up.
        """
        with self._state_lock:
            return self._state == _OrderedSequencerStatus.FINISHED

    def stop(self) -> None:
        """Permanently stop this sequencer.

        This differs from pausing, which may be resumed. Immediately commits
        the first batch and cancels the rest.

        Raises:
            RuntimeError:
                If called after stop() has already been called.
        """
        with self._state_lock:
            if self._state == _OrderedSequencerStatus.STOPPED:
                raise RuntimeError('Ordered sequencer already stopped.')
            self._state = _OrderedSequencerStatus.STOPPED
            if self._ordered_batches:
                self._ordered_batches[0].commit()
                while len(self._ordered_batches) > 1:
                    batch = self._ordered_batches.pop()
                    batch.cancel(batch_base.BatchCancellationReason.CLIENT_STOPPED)

    def commit(self) -> None:
        """Commit the first batch, if unpaused.

        If paused or no batches exist, this method does nothing.

        Raises:
            RuntimeError:
                If called after stop() has already been called.
        """
        with self._state_lock:
            if self._state == _OrderedSequencerStatus.STOPPED:
                raise RuntimeError('Ordered sequencer already stopped.')
            if self._state != _OrderedSequencerStatus.PAUSED and self._ordered_batches:
                self._ordered_batches[0].commit()

    def _batch_done_callback(self, success: bool) -> None:
        """Deal with completion of a batch.

        Called when a batch has finished publishing, with either a success
        or a failure. (Temporary failures are retried infinitely when
        ordering keys are enabled.)
        """
        ensure_cleanup_and_commit_timer_runs = False
        with self._state_lock:
            assert self._state != _OrderedSequencerStatus.PAUSED, 'This method should not be called after pause() because pause() should have cancelled all of the batches.'
            assert self._state != _OrderedSequencerStatus.FINISHED, 'This method should not be called after all batches have been finished.'
            self._ordered_batches.popleft()
            if success:
                if len(self._ordered_batches) == 0:
                    self._state = _OrderedSequencerStatus.FINISHED
                    ensure_cleanup_and_commit_timer_runs = True
                elif len(self._ordered_batches) == 1:
                    ensure_cleanup_and_commit_timer_runs = True
                else:
                    self._ordered_batches[0].commit()
            else:
                self._pause()
        if ensure_cleanup_and_commit_timer_runs:
            self._client.ensure_cleanup_and_commit_timer_runs()

    def _pause(self) -> None:
        """Pause this sequencer: set state to paused, cancel all batches, and
        clear the list of ordered batches.

        _state_lock must be taken before calling this method.
        """
        assert self._state != _OrderedSequencerStatus.FINISHED, 'Pause should not be called after all batches have finished.'
        self._state = _OrderedSequencerStatus.PAUSED
        for batch in self._ordered_batches:
            batch.cancel(batch_base.BatchCancellationReason.PRIOR_ORDERED_MESSAGE_FAILED)
        self._ordered_batches.clear()

    def unpause(self) -> None:
        """Unpause this sequencer.

        Raises:
            RuntimeError:
                If called when the ordering key has not been paused.
        """
        with self._state_lock:
            if self._state != _OrderedSequencerStatus.PAUSED:
                raise RuntimeError('Ordering key is not paused.')
            self._state = _OrderedSequencerStatus.ACCEPTING_MESSAGES

    def _create_batch(self, commit_retry: 'OptionalRetry'=gapic_v1.method.DEFAULT, commit_timeout: 'types.OptionalTimeout'=gapic_v1.method.DEFAULT) -> '_batch.thread.Batch':
        """Create a new batch using the client's batch class and other stored
            settings.

        Args:
            commit_retry:
                The retry settings to apply when publishing the batch.
            commit_timeout:
                The timeout to apply when publishing the batch.
        """
        return self._client._batch_class(client=self._client, topic=self._topic, settings=self._client.batch_settings, batch_done_callback=self._batch_done_callback, commit_when_full=False, commit_retry=commit_retry, commit_timeout=commit_timeout)

    def publish(self, message: gapic_types.PubsubMessage, retry: 'OptionalRetry'=gapic_v1.method.DEFAULT, timeout: 'types.OptionalTimeout'=gapic_v1.method.DEFAULT) -> futures.Future:
        """Publish message for this ordering key.

        Args:
            message:
                The Pub/Sub message.
            retry:
                The retry settings to apply when publishing the message.
            timeout:
                The timeout to apply when publishing the message.

        Returns:
            A class instance that conforms to Python Standard library's
            :class:`~concurrent.futures.Future` interface (but not an
            instance of that class). The future might return immediately with a
            PublishToPausedOrderingKeyException if the ordering key is paused.
            Otherwise, the future tracks the lifetime of the message publish.

        Raises:
            RuntimeError:
                If called after this sequencer has been stopped, either by
                a call to stop() or after all batches have been published.
        """
        with self._state_lock:
            if self._state == _OrderedSequencerStatus.PAUSED:
                errored_future = futures.Future()
                exception = exceptions.PublishToPausedOrderingKeyException(self._ordering_key)
                errored_future.set_exception(exception)
                return errored_future
            if self._state == _OrderedSequencerStatus.FINISHED:
                self._state = _OrderedSequencerStatus.ACCEPTING_MESSAGES
            if self._state == _OrderedSequencerStatus.STOPPED:
                raise RuntimeError('Cannot publish on a stopped sequencer.')
            assert self._state == _OrderedSequencerStatus.ACCEPTING_MESSAGES, 'Publish is only allowed in accepting-messages state.'
            if not self._ordered_batches:
                new_batch = self._create_batch(commit_retry=retry, commit_timeout=timeout)
                self._ordered_batches.append(new_batch)
            batch = self._ordered_batches[-1]
            future = batch.publish(message)
            while future is None:
                batch = self._create_batch(commit_retry=retry, commit_timeout=timeout)
                self._ordered_batches.append(batch)
                future = batch.publish(message)
            return future

    def _set_batch(self, batch: '_batch.thread.Batch') -> None:
        self._ordered_batches = collections.deque([batch])

    def _set_batches(self, batches: Iterable['_batch.thread.Batch']) -> None:
        self._ordered_batches = collections.deque(batches)

    def _get_batches(self) -> Sequence['_batch.thread.Batch']:
        return self._ordered_batches