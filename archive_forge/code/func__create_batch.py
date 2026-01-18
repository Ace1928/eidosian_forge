import typing
from typing import Optional
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher._sequencer import base
from google.pubsub_v1 import types as gapic_types
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